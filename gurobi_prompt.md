# Context Prompt for Gurobi Implementation
# Paste this at the start of any LLM conversation before asking for notebook code.

---

## PROJECT OVERVIEW

I am building a cooperative game theory pipeline for carbon farming by smallholder farmers in India. The pipeline has four Jupyter notebooks. I need help implementing each one using Gurobi as the solver.

The goal is:
1. Find each farmer's optimal carbon farming practice portfolio (solo)
2. Find the optimal joint portfolio for every possible coalition of farmers
3. Distribute the coalition's surplus fairly using game-theoretic allocation rules

---

## FILES I HAVE from data generator:
data_genearto.py:
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

farmer_generator.py:
# -*- coding: utf-8 -*-
"""farmers_generator.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UQcKexElXFVqXB1j_wT6XtB1hXb5oMGT
"""

# farmers_generator.py
# Generates farmers.csv for SALM / carbon-coalition experiments
#
# Spec (editable):
# - n farmers
# - farm size ~ Lognormal(mu, sigma), truncated to [min_ha, max_ha], rounded
# - budget proportional to size: (budget_per_ha * (1 ± noise)) * size
#
# Output columns:
# Farmer_ID, Farm_Size_ha, Budget_per_ha_INR_per_season, Budget_total_INR_per_season

import numpy as np
import pandas as pd
from pathlib import Path


def truncated_lognormal(
    rng: np.random.Generator,
    n: int,
    mu: float,
    sigma: float,
    low: float,
    high: float,
    round_decimals: int = 2,
    batch_size: int = 10000,
) -> np.ndarray:
    """Sample n values from Lognormal(mu, sigma), truncated to [low, high]."""
    vals = []
    while len(vals) < n:
        x = rng.lognormal(mean=mu, sigma=sigma, size=batch_size)
        x = x[(x >= low) & (x <= high)]
        vals.extend(x.tolist())

    x = np.array(vals[:n])
    x = np.round(x, round_decimals)
    # guardrail after rounding
    x = np.clip(x, low, high)
    x = np.round(x, round_decimals)
    return x


def generate_farmers_df(
    n_farmers: int = 15,
    seed: int = 42,
    mu: float = -0.5,
    sigma: float = 1.0,
    min_ha: float = 0.20,
    max_ha: float = 2.00,
    round_decimals: int = 2,
    budget_per_ha: float = 3500.0,          # INR/ha/season
    budget_noise_pct: float = 0.10,         # ±10%
) -> pd.DataFrame:
    """Return a farmers DataFrame with farm size + budget."""
    rng = np.random.default_rng(seed)

    farm_size = truncated_lognormal(
        rng=rng,
        n=n_farmers,
        mu=mu,
        sigma=sigma,
        low=min_ha,
        high=max_ha,
        round_decimals=round_decimals,
    )

    # Budget per ha with uniform ±noise
    noise = rng.uniform(-budget_noise_pct, budget_noise_pct, size=n_farmers)
    budget_per_ha_realized = budget_per_ha * (1.0 + noise)
    budget_total = farm_size * budget_per_ha_realized

    df = pd.DataFrame({
        "Farmer_ID": [f"F{i+1:04d}" for i in range(n_farmers)],
        "Farm_Size_ha": farm_size,
        "Budget_per_ha_INR_per_season": np.round(budget_per_ha_realized, 2),
        "Budget_total_INR_per_season": np.round(budget_total, 2),
    })

    return df


def main():
    # ---- Edit parameters here if needed ----
    params = dict(
        n_farmers=15,
        seed=42,
        mu=-0.5,
        sigma=1.0,
        min_ha=0.20,
        max_ha=2.00,
        round_decimals=2,
        budget_per_ha=3500.0,
        budget_noise_pct=0.10,
    )

    out_dir = Path("synthetic_salm_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_farmers_df(**params)

    out_path = out_dir / "farmers.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path.resolve()}")
    print("\nFarm size summary (ha):")
    print(df["Farm_Size_ha"].describe().round(3))
    print("\nBudget total summary (INR/season):")
    print(df["Budget_total_INR_per_season"].describe().round(2))


if __name__ == "__main__":
    main()

### Farmer data (two CSVs, same structure)
- `data/Dataset_INR/15_farmers.csv` — 15 farmers
- `data/Dataset_INR/25_farmers.csv` — 25 farmers

Columns:
- `Farmer_ID` : string (F0001 … F0025)
- `Farm_Size_ha` : float, farm size in hectares
- `Budget_per_ha_INR_per_season` : float
- `Budget_total_INR_per_season` : float  ← this is B_i, the binding budget constraint

### Practice data
- `data/Dataset_INR/practices.csv`

Columns:
- `Practice` : string, practice name (20 practices total)
- `Net_CSP_base` : float, carbon sequestration potential in tCO2e/ha/season
- `Net_OC_per_ha` : float, operational cost in INR/ha/season
  - NEGATIVE means the practice saves money vs conventional
  - POSITIVE means it costs extra money
- `Base_yield_change` : float, change in yield in tons/ha/season
  - POSITIVE = yield gain, NEGATIVE = yield loss

### Interaction matrices (all 20×20, symmetric, zero diagonal)
- `data/Dataset_INR/alpha_carbon.csv`
  - α_jk : tCO2e/ha/season added to sequestration when practices j and k are both adopted
- `data/Dataset_INR/beta_cost.csv`
  - β_jk : INR/ha/season added to cost when practices j and k are both adopted
  - Negative = cost saving synergy
- `data/Dataset_INR/gamma_yield.csv`
  - γ_jk : tons/ha/season added to yield when practices j and k are both adopted
- `data/Dataset_INR/delta_incompatibility.csv`
  - Δ_jk : 1 if practices j and k CANNOT be co-adopted, 0 if compatible
  - Must be enforced as a hard constraint: x_j + x_k ≤ 1 for all pairs where Δ_jk = 1

---

## ECONOMIC PARAMETERS (i will have to play with this)

```python
CCP           = 1500    # INR / tCO2e       carbon credit price
PADDY_PRICE   = 22_000  # INR / ton         farm-gate paddy price
Fixed_MRV     = 5_000   # INR               flat MRV cost per certification
Variable_MRV  = 2_000   # INR / ha^delta    scales with total coalition area
delta_mrv     = 0.7     # exponent          (<1 = economies of scale in MRV)
Fixed_T       = 2_000   # INR               flat transaction/registry fee
Variable_T    = 500     # INR / farmer      per-member transaction cost
```

These are placeholder values. so
they should be changed in one place.

---

## MODEL FORMULATION

### Decision variables
For farmer i, practice j:
- x_ij ∈ {0, 1} — binary: does farmer i adopt practice j?

### Portfolio-level quantities (for farmer i with portfolio Π_i)

**Sequestration** (tCO2e/season):
```
CSP_i(x_i) = FS_i * [ Σ_j (Net_CSP_base_j * x_ij)
                     + Σ_{j<k} (α_jk * x_ij * x_ik) ]
```

**Operational cost** (INR/season):
```
OC_i(x_i) = FS_i * [ Σ_j (Net_OC_per_ha_j * x_ij)
                    + Σ_{j<k} (β_jk * x_ij * x_ik) ]
```

**Yield revenue** (INR/season):
```
YR_i(x_i) = FS_i * PADDY_PRICE * [ Σ_j (Base_yield_change_j * x_ij)
                                   + Σ_{j<k} (γ_jk * x_ij * x_ik) ]
```

**Note on pairwise terms:** x_ij * x_ik is a product of two binary variables.
Linearise using a new binary variable z_ijk = x_ij * x_ik with constraints:
  z_ijk ≤ x_ij
  z_ijk ≤ x_ik
  z_ijk ≥ x_ij + x_ik - 1
  z_ijk ∈ {0, 1}

### MRV and transaction costs

**Solo farmer i:**
```
C_MRV_solo(i) = Fixed_MRV + Variable_MRV * FS_i^delta_mrv
C_T_solo      = Fixed_T + Variable_T          (|S|=1 so Variable_T * 1)
```

**Coalition S:**
```
C_MRV_coalition(S) = Fixed_MRV + Variable_MRV * (Σ_{i∈S} FS_i)^delta_mrv
C_T_coalition(S)   = Fixed_T + Variable_T * |S|
```

---

## NOTEBOOK 01 — Data Ingestion and Preprocessing

Tasks:
1. Load all CSVs into pandas DataFrames
2. Load config.py parameters
3. Build numpy arrays for Alpha, Beta, Gamma, Delta indexed by practice name
4. Validate: check all matrices are symmetric, Delta diagonal is zero,
   no NaN values anywhere
5. For each incompatible pair (Delta_jk = 1), print the pair names
6. Save processed inputs as a pickle: `data/processed/optimization_inputs.pkl`
   containing a dict with keys: farmers_df, practices_df, Alpha, Beta, Gamma,
   Delta, practice_names, config

No Gurobi needed in this notebook.

---

## NOTEBOOK 02 — Single Farmer Problem

For each farmer i independently, solve:

**Objective** (maximise net profit):
```
max  FS_i * CSP_i(x_i) * CCP
   + YR_i(x_i)
   - OC_i(x_i)
   - C_MRV_solo(i)
   - C_T_solo
```

**Constraints:**
1. Budget: OC_i(x_i) + C_MRV_solo(i) + C_T_solo ≤ B_i
2. Incompatibility: x_ij + x_ik ≤ 1 for all (j,k) where Δ_jk = 1
3. Binary: x_ij ∈ {0,1}
4. Linearisation of pairwise terms (z_ijk as above)

**Outputs per farmer:**
- Optimal portfolio (list of practice names adopted)
- Standalone value ṽ({i}) = optimal objective value
  - If the problem is infeasible (budget too small even for empty portfolio
    after fixed MRV/T costs), set ṽ({i}) = 0 and portfolio = []
- Carbon sequestration achieved (tCO2e/season)
- Operational cost (INR/season)
- Yield revenue change (INR/season)

Save results as `data/processed/standalone_values.pkl` — a dict keyed by
Farmer_ID containing the above fields.

Also print a summary table showing all 15 (or 25) farmers with their
standalone values ranked highest to lowest.

---

## NOTEBOOK 03 — Coalition Problem

For every subset S ⊆ {F1…F15} (2^15 = 32768 subsets for 15 farmers):

**Joint optimisation** — each farmer i ∈ S chooses their own portfolio x_i:

**Objective:**
```
max  Σ_{i∈S} [ FS_i * CSP_i(x_i) * CCP + YR_i(x_i) - OC_i(x_i) ]
   - C_MRV_coalition(S)
   - C_T_coalition(S)
```

**Constraints per farmer i ∈ S:**
1. Individual budget: OC_i(x_i) ≤ B_i  (each farmer pays their own OC)
2. Incompatibility: x_ij + x_ik ≤ 1 for all (j,k) where Δ_jk = 1
3. Binary + linearisation as before

**Coalition feasibility (CF1 — pooled residual):**
```
Σ_{i∈S} (B_i - OC_i(x_i)) ≥ C_MRV_coalition(S) + C_T_coalition(S)
```
This ensures the coalition collectively can afford shared certification costs.

**Coalition value:**
```
v(S) = optimal objective value if feasible, else v(S) = 0
```

Note: For |S| = 1, v({i}) should equal ṽ({i}) from Notebook 02. Verify this.

**Implementation notes:**
- 32768 Gurobi solves is feasible but slow. Use multiprocessing or joblib
  to parallelise. Show a progress bar (tqdm).
- Each solve is small (15*20 = 300 binary variables max). Should be fast.
- Store results in a dict: characteristic_function[frozenset(S)] = v(S)
- Save as `data/processed/characteristic_function.pkl`

---

## NOTEBOOK 04 — Surplus Sharing

Load characteristic_function.pkl and standalone_values.pkl.

Let S = grand coalition (all farmers). Compute five allocation mechanisms.

**Allocation vector x = (x_i)_{i∈S}** where x_i is farmer i's payoff.

### 1. Core check + Least Core (LP)

Variables: x_i for each farmer, ε (scalar slack)

Constraints:
- Efficiency:           Σ_{i∈S} x_i = v(S)
- Individual rationality: x_i ≥ ṽ({i})  for all i
- Sub-coalition rationality: Σ_{i∈T} x_i ≥ v(T) - ε  for all T ⊂ S, T ≠ ∅

First solve with ε = 0. If feasible → core is non-empty, report it.
If infeasible → solve minimising ε (least core):
```
min ε
subject to above constraints, ε ≥ 0
```
Report ε* (residual instability measure).

### 2. Nucleolus

Computed as a sequence of LPs. At each iteration:
- Minimise the maximum excess e(T, x) = v(T) - Σ_{i∈T} x_i
- Fix constraints that are tight (within tolerance) and proceed
- Stop when all excesses are determined

### 3. Shapley Value

```
φ_i = Σ_{T ⊆ S\{i}} [ |T|! * (|S|-|T|-1)! / |S|! ] * (v(T∪{i}) - v(T))
```
Compute directly from the characteristic function table (no new Gurobi solves).
After computing, verify: Σ φ_i = v(S) and φ_i ≥ ṽ({i}) for all i.

### 4. Equal Split
```
x_i = v(S) / |S|
```

### 5. Proportional Split
```
x_i = [ FS_i * CSP_i(Π_i^co) ] / [ Σ_{j∈S} FS_j * CSP_j(Π_j^co) ] * v(S)
```
where Π_i^co is farmer i's portfolio in the grand coalition (from Notebook 03).

### Output
Print a comparison table: all 5 mechanisms × all farmers.
For each mechanism report:
- Whether individual rationality holds for every farmer
- Whether core constraints are satisfied (or ε* for least core)

---

## IMPORTANT IMPLEMENTATION NOTES

1. **Linearisation is mandatory.** The pairwise interaction terms (α_jk * x_ij * x_ik)
   make the objective nonlinear. You MUST introduce z_ijk = x_ij * x_ik and
   use the linearisation constraints. Gurobi can handle this as a MILP.

2. **Incompatibility as constraint, not penalty.** Never use -inf values.
   Always enforce incompatibility as: x_ij + x_ik ≤ 1.

3. **Empty portfolio is always feasible.** If a farmer adopts no practices,
   OC = 0, CSP = 0, yield change = 0. Net profit = -C_MRV_solo - C_T_solo.
   This may be negative, meaning solo participation is not worthwhile.
   This is a valid and important result — it motivates coalition formation.

4. **Sign convention for OC:** Net_OC_per_ha can be negative (ZT saves money).
   Total OC_i = FS_i * Σ_j(Net_OC_per_ha_j * x_ij) + FS_i * Σ_{j<k}(β_jk * z_ijk)
   This can be negative overall (net cost saving). The budget constraint
   OC_i ≤ B_i is still valid — a negative OC always satisfies it.

5. **Gurobi model cleanup.** Call model.dispose() after each solve in the
   coalition enumeration loop to avoid memory buildup over 32768 solves.
