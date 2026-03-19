"""
data_generator.py
=================
Generates all practice-level and farmer-level data needed for the
carbon farming coalition optimisation pipeline.

Run this script once before any notebook. All outputs are saved to
data/Dataset_INR/. The two farmer CSVs (15_farmers.csv, 25_farmers.csv)
are assumed to already exist there and are not modified.

Outputs
-------
practices.csv
    Per-practice base data: CSP, OC, yield change. One row per practice.

alpha_carbon.csv       (20 x 20)
    Carbon sequestration interaction matrix, α_jk.
    Units: tCO2e / ha / season.
    Positive = synergy, negative = antagonism, 0 = incompatible or diagonal.

beta_cost.csv          (20 x 20)
    Operational cost interaction matrix, β_jk.
    Units: INR / ha / season.
    Negative = cost saving synergy, positive = cost burden.

gamma_yield.csv        (20 x 20)
    Yield interaction matrix, γ_jk.
    Units: tons / ha / season.
    Positive = extra yield gain, negative = extra yield loss.
    Multiply by PADDY_PRICE_INR_PER_TON to convert to revenue in the optimizer.

delta_incompatibility.csv  (20 x 20)
    Binary incompatibility matrix, Δ_jk.
    1 = practices j and k cannot be co-adopted (enforced as hard constraint).
    0 = compatible.

farmer_csp.csv         (n_farmers x 20)
    Farmer-specific realised sequestration, CSP_ij.
    Units: tCO2e / ha / season.
    Currently set equal to the uniform base (CSP_ij = CSP_j for all i).
    Replace this file with measured/modelled heterogeneous values if available.

Sign conventions (used consistently throughout all notebooks)
-------------------------------------------------------------
- Net_OC_per_ha  : positive = costs money, negative = saves money vs conventional.
                   Applied as OC_j(FS_i) = Net_OC_per_ha_j * FS_i  (linear).
- Base_yield_change : positive = yield gain (farmer earns more),
                      negative = yield loss.
                   Applied as yield_revenue_i = FS_i * Y(Pi) * PADDY_PRICE
                   and ADDED to the objective (not subtracted).
- β_jk           : scaled by FS_i in the optimizer (consistent with linear OC).
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("data/Dataset_INR")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global economic constant (used only to document scale; not embedded) ──────
PADDY_PRICE_INR_PER_TON = 22_000   # INR per ton of paddy — for reference only.
                                    # The optimizer multiplies gamma by this.

# =============================================================================
# 1. PRACTICES — base data
# =============================================================================
# Net_OC_per_ha : INR / ha / season
#   Negative values mean the practice saves money vs. conventional tillage.
#   Applied linearly: total_OC_j = Net_OC_per_ha_j * FS_i
#
# Net_CSP_base  : tCO2e / ha / season (uniform across farmers)
#
# Base_yield_change : tons / ha / season
#   Positive = net yield gain from adopting this practice alone.
#   Converted to INR in the optimizer: revenue = FS_i * yield_change * 22000.

practices_data = {
    # Practice name                                    : (Net_CSP_base, Net_OC_per_ha, Base_yield_change)
    "Zero Tillage (ZT)":                               (1.20,  -3500,  0.05),
    "Crop Residue Retention":                          (1.10,  -3500,  0.15),
    "Reduced / Minimum Tillage":                       (0.32,  -2000,  0.00),
    "Puddling Reduction":                              (0.25,  -2000, -0.10),
    "Mid-Season Drainage (MSD)":                       (0.40,  -2000, -0.10),
    "Alternate Wetting and Drying (AWD)":              (0.55,  -2000,  0.00),
    "System of Rice Intensification (SRI) Water Management": (0.60, 0,  0.30),
    "Balanced Inorganic NPK Fertilisation":            (0.25,      0,  0.20),
    "Optimised / Variable-Rate N Application":         (0.35,   3000,  0.10),
    "Integrated Nutrient Management (INM) – Organic + Inorganic": (0.85, 0, 0.10),
    "Green Manure Incorporation":                      (0.45,  -2000,  0.20),
    "Farmyard Manure (FYM) Application":               (0.33,      0,  0.10),
    "Compost Application":                             (0.50,   3000,  0.12),
    "Biochar Application":                             (2.20,   8000,  0.30),
    "Rice Straw Incorporation with N Fertiliser":      (0.95,  -2000,  0.20),
    "Crop Rotation and Diversification":               (1.10,      0,  0.15),
    "Early-Maturing / Low-Emission Variety Selection": (0.30,  -2000,  0.00),
    "High-Biomass / Deep-Root Variety Selection":      (0.70,  -2000,  0.20),
    "Deep Placement of N Fertiliser":                  (0.30,      0,  0.25),
    "Raised Bed + Zero Tillage System":                (2.50,   3000,  0.10),
}

practices = list(practices_data.keys())
n = len(practices)
idx = {p: i for i, p in enumerate(practices)}

practices_df = pd.DataFrame(
    [
        {
            "Practice":           p,
            "Net_CSP_base":       v[0],
            "Net_OC_per_ha":      v[1],
            "Base_yield_change":  v[2],
        }
        for p, v in practices_data.items()
    ]
)
practices_df.to_csv(OUT_DIR / "practices.csv", index=False)
print("Saved: practices.csv")

# =============================================================================
# 2. INTERACTION MATRICES — α, β, γ, Δ
# =============================================================================

SEED = 42
rng = np.random.default_rng(SEED)

Alpha = np.zeros((n, n), dtype=float)
Beta  = np.zeros((n, n), dtype=float)
Gamma = np.zeros((n, n), dtype=float)
Delta = np.zeros((n, n), dtype=int)

# ── 2a. Pair categories ───────────────────────────────────────────────────────
pair_category = {}
pair_reason   = {}

def set_cat(p1, p2, cat, reason):
    key = tuple(sorted((p1, p2)))
    pair_category[key] = cat
    pair_reason[key]   = reason

# Incompatible pairs (hard constraint — Δ_jk = 1)
for p1, p2, reason in [
    (
        "Zero Tillage (ZT)",
        "Raised Bed + Zero Tillage System",
        "Raised Bed + ZT subsumes ZT; co-adoption double-counts sequestration.",
    ),
    (
        "Balanced Inorganic NPK Fertilisation",
        "Integrated Nutrient Management (INM) – Organic + Inorganic",
        "INM already includes balanced inorganic NPK; co-adoption double-counts.",
    ),
]:
    set_cat(p1, p2, "X", reason)

# Strong positive pairs
for p1, p2, reason in [
    ("Zero Tillage (ZT)", "Crop Residue Retention",
     "Core conservation-agriculture synergy; residue mulch preserved under undisturbed soil."),
    ("Zero Tillage (ZT)", "Integrated Nutrient Management (INM) – Organic + Inorganic",
     "ZT protects soil aggregates while INM replenishes active carbon pools."),
    ("Crop Residue Retention", "Alternate Wetting and Drying (AWD)",
     "AWD mitigates the CH4 amplification that residue retention creates under full flooding."),
    ("Mid-Season Drainage (MSD)", "Biochar Application",
     "Biochar compensates SOC loss from drainage while MSD suppresses CH4."),
    ("Alternate Wetting and Drying (AWD)", "Early-Maturing / Low-Emission Variety Selection",
     "Shorter submergence period amplifies AWD's CH4-reduction benefit."),
    ("Balanced Inorganic NPK Fertilisation", "Farmyard Manure (FYM) Application",
     "NPK + FYM improves labile C and nutrient use efficiency."),
    ("Integrated Nutrient Management (INM) – Organic + Inorganic", "Green Manure Incorporation",
     "Green manure sustains active and passive C pools alongside INM."),
    ("System of Rice Intensification (SRI) Water Management",
     "Integrated Nutrient Management (INM) – Organic + Inorganic",
     "SRI × INM: high-efficiency combination for soil health, yield, and CSP."),
]:
    if tuple(sorted((p1, p2))) not in pair_category:
        set_cat(p1, p2, "strong_pos", reason)

# Moderate positive pairs
for p1, p2, reason in [
    ("Reduced / Minimum Tillage", "Crop Residue Retention",
     "Reduced tillage preserves residue-derived SOC."),
    ("Puddling Reduction", "Zero Tillage (ZT)",
     "Both reduce soil structural disturbance and protect SOC."),
    ("Puddling Reduction", "Crop Residue Retention",
     "Residue retention plus less aggressive wet tillage lowers GHG burden."),
    ("Mid-Season Drainage (MSD)", "Optimised / Variable-Rate N Application",
     "Named synergy for net GHG balance under drainage."),
    ("System of Rice Intensification (SRI) Water Management",
     "Alternate Wetting and Drying (AWD)",
     "SRI and AWD-type water control are synergistic."),
    ("System of Rice Intensification (SRI) Water Management",
     "Farmyard Manure (FYM) Application",
     "SRI synergistic with organic amendments."),
    ("System of Rice Intensification (SRI) Water Management", "Compost Application",
     "SRI synergistic with organic amendments."),
    ("System of Rice Intensification (SRI) Water Management", "Green Manure Incorporation",
     "SRI synergistic with organic amendments."),
    ("System of Rice Intensification (SRI) Water Management",
     "Early-Maturing / Low-Emission Variety Selection",
     "SRI synergistic with improved varieties."),
    ("Balanced Inorganic NPK Fertilisation", "Deep Placement of N Fertiliser",
     "Improved N placement enhances balanced nutrient management."),
    ("Optimised / Variable-Rate N Application", "Deep Placement of N Fertiliser",
     "Both improve N-use efficiency and reduce losses."),
    ("Biochar Application", "Deep Placement of N Fertiliser",
     "Biochar and deep N placement jointly improve nutrient retention."),
    ("Crop Rotation and Diversification", "Compost Application",
     "Diversified systems and organic amendments reinforce soil-carbon building."),
    ("Crop Rotation and Diversification", "Green Manure Incorporation",
     "Green manure fits naturally into diversified rotations."),
    ("Crop Rotation and Diversification", "Farmyard Manure (FYM) Application",
     "Diversification and organic matter addition are complementary."),
    ("Crop Rotation and Diversification",
     "Integrated Nutrient Management (INM) – Organic + Inorganic",
     "Diversified systems and INM reinforce nutrient efficiency."),
    ("Rice Straw Incorporation with N Fertiliser",
     "Optimised / Variable-Rate N Application",
     "Better N timing improves residue decomposition management."),
    ("Rice Straw Incorporation with N Fertiliser", "Deep Placement of N Fertiliser",
     "Improved N placement supports residue incorporation efficiency."),
    ("High-Biomass / Deep-Root Variety Selection",
     "Alternate Wetting and Drying (AWD)",
     "Positive interaction with AWD-type water management."),
]:
    if tuple(sorted((p1, p2))) not in pair_category:
        set_cat(p1, p2, "pos", reason)

# Moderate negative pairs
for p1, p2, reason in [
    ("Alternate Wetting and Drying (AWD)", "Biochar Application",
     "Biochar + AWD may increase priming of native SOM."),
    ("High-Biomass / Deep-Root Variety Selection", "Mid-Season Drainage (MSD)",
     "High-biomass varieties are antagonistic with drainage-type water regimes."),
]:
    if tuple(sorted((p1, p2))) not in pair_category:
        set_cat(p1, p2, "neg", reason)

# ── 2b. Sampling distributions ────────────────────────────────────────────────
# Alpha: tCO2e/ha/season  — interaction is a second-order effect on top of base CSP (0.25–2.5)
# Beta : INR/ha/season    — scaled by FS_i in the optimizer (consistent with linear OC)
# Gamma: tons/ha/season   — second-order effect; base yield changes are 0.00–0.30 t/ha

dist_params = {
    "Alpha": {
        "strong_pos": {"mean":  0.16, "sd": 0.04},
        "pos":        {"mean":  0.08, "sd": 0.03},
        "neg":        {"mean": -0.07, "sd": 0.02},
        "zero_like":  {"mean":  0.00, "sd": 0.015},
    },
    "Beta": {
        # Negative = cost-saving synergy, Positive = extra cost burden
        "strong_pos": {"mean": -220.0, "sd":  50.0},
        "pos":        {"mean":  -90.0, "sd":  35.0},
        "neg":        {"mean":  120.0, "sd":  40.0},
        "zero_like":  {"mean":    0.0, "sd":  25.0},
    },
    "Gamma": {
        "strong_pos": {"mean":  0.10, "sd": 0.03},
        "pos":        {"mean":  0.05, "sd": 0.02},
        "neg":        {"mean": -0.05, "sd": 0.02},
        "zero_like":  {"mean":  0.00, "sd": 0.01},
    },
}

ALPHA_CLIP = (-0.30,  0.30)
BETA_CLIP  = (-500.0, 300.0)
GAMMA_CLIP = (-0.15,  0.15)

# ── 2c. Fill matrices ─────────────────────────────────────────────────────────
for p1, p2 in combinations(practices, 2):
    key = tuple(sorted((p1, p2)))
    cat = pair_category.get(key, "zero_like")
    i, j = idx[p1], idx[p2]

    if cat == "X":
        Delta[i, j] = Delta[j, i] = 1
        # α, β, γ remain 0 — incompatible pairs never contribute to the objective
    else:
        a = float(np.clip(rng.normal(dist_params["Alpha"][cat]["mean"],
                                     dist_params["Alpha"][cat]["sd"]),  *ALPHA_CLIP))
        b = float(np.clip(rng.normal(dist_params["Beta"][cat]["mean"],
                                     dist_params["Beta"][cat]["sd"]),   *BETA_CLIP))
        g = float(np.clip(rng.normal(dist_params["Gamma"][cat]["mean"],
                                     dist_params["Gamma"][cat]["sd"]),  *GAMMA_CLIP))
        Alpha[i, j] = Alpha[j, i] = a
        Beta[i, j]  = Beta[j, i]  = b
        Gamma[i, j] = Gamma[j, i] = g

np.fill_diagonal(Alpha, 0.0)
np.fill_diagonal(Beta,  0.0)
np.fill_diagonal(Gamma, 0.0)
np.fill_diagonal(Delta, 0)

# ── 2d. Sanity checks ─────────────────────────────────────────────────────────
assert np.allclose(Alpha, Alpha.T), "Alpha not symmetric"
assert np.allclose(Beta,  Beta.T),  "Beta not symmetric"
assert np.allclose(Gamma, Gamma.T), "Gamma not symmetric"
assert np.array_equal(Delta, Delta.T), "Delta not symmetric"
print("Matrix symmetry checks passed.")

# ── 2e. Save matrices ─────────────────────────────────────────────────────────
pd.DataFrame(Alpha, index=practices, columns=practices).to_csv(OUT_DIR / "alpha_carbon.csv")
pd.DataFrame(Beta,  index=practices, columns=practices).to_csv(OUT_DIR / "beta_cost.csv")
pd.DataFrame(Gamma, index=practices, columns=practices).to_csv(OUT_DIR / "gamma_yield.csv")
pd.DataFrame(Delta, index=practices, columns=practices).to_csv(OUT_DIR / "delta_incompatibility.csv")
print("Saved: alpha_carbon.csv, beta_cost.csv, gamma_yield.csv, delta_incompatibility.csv")

# =============================================================================
# 3. QUICK SUMMARY
# =============================================================================
print("\n── Data generation complete ──────────────────────────────────────────")
print(f"  Practices           : {n}")
print(f"  Incompatible pairs  : {int(Delta.sum() / 2)}")
print(f"  Strong-pos pairs    : {sum(1 for v in pair_category.values() if v == 'strong_pos')}")
print(f"  Moderate-pos pairs  : {sum(1 for v in pair_category.values() if v == 'pos')}")
print(f"  Negative pairs      : {sum(1 for v in pair_category.values() if v == 'neg')}")
print(f"  Zero-like pairs     : {sum(1 for v in pair_category.values() if v == 'zero_like')}")
print(f"\nFiles written to: {OUT_DIR.resolve()}")
print("  practices.csv")
print("  alpha_carbon.csv")
print("  beta_cost.csv")
print("  gamma_yield.csv")
print("  delta_incompatibility.csv")
print("\nNOTE: CSP_ij = CSP_j for all farmers (uniform sequestration).")
print("      The optimizer reads CSP_j directly from practices.csv.")
print("      If soil survey data becomes available, add a farmer_csp.csv")
print("      and update the optimizer to read it instead.")
print("\nStill needed before running the optimizer (define in config.py):")
print("  CCP          — carbon credit price       (INR / tCO2e)")
print("  Fixed_MRV    — flat MRV cost             (INR)")
print("  Variable_MRV — variable MRV rate         (INR / ha^delta)")
print("  delta        — MRV scale exponent        (suggested: 0.6–0.8)")
print("  Fixed_T      — flat transaction cost     (INR)")
print("  Variable_T   — per-farmer transaction cost (INR / farmer)")