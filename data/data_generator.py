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

alpha_carbon.csv       (12 x 12)
    Carbon sequestration interaction matrix, α_jk.
    Units: tCO2e / ha / season.
    Positive = synergy, negative = antagonism, 0 = diagonal or zero-like.

beta_cost.csv          (12 x 12)
    Operational cost interaction matrix, β_jk.
    Units: INR / ha / season.
    Negative = cost-saving synergy, positive = cost burden.

gamma_yield.csv        (12 x 12)
    Yield interaction matrix, γ_jk.
    Units: tons / ha / season.
    Positive = extra yield gain, negative = extra yield loss.
    Multiply by PADDY_PRICE_INR_PER_TON to convert to revenue in the optimizer.

delta_incompatibility.csv  (12 x 12)
    Binary incompatibility matrix, Δ_jk.
    1 = practices j and k cannot be co-adopted (enforced as hard constraint).
    0 = compatible.

farmer_csp.csv         (n_farmers x 12)
    Farmer-specific realised sequestration, CSP_ij.
    Units: tCO2e / ha / season.
    Currently set equal to the uniform base (CSP_ij = CSP_j for all i).

Sign conventions (used consistently throughout all notebooks)
-------------------------------------------------------------
- Net_OC_per_ha       : positive = costs money, negative = saves money.
- Base_yield_change   : positive = yield gain, negative = yield loss.
- β_jk (cost)         : pos category  → mean is negative INR (saving synergy)
                        neg category  → mean is positive INR (cost burden)
                        zero_like     → mean ≈ 0 (default for unlisted pairs)
- α_jk, γ_jk          : pos = synergy, neg = antagonism, zero_like = default.
Each matrix is sampled independently from its own per-dimension category.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("data/Dataset_INR")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PADDY_PRICE_INR_PER_TON = 22_000   # INR per ton — used by the optimizer externally.

# =============================================================================
# 1. PRACTICES — base data
# =============================================================================

practices_data = {
    "Crop Residue Retention":                                       (1.10,  -3500,  0.15),
    "Reduced / Minimum Tillage":                                    (0.32,  -2000, -0.10),
    "Mid-Season Drainage (MSD)":                                    (0.40,  -2000, -0.10),
    "Alternate Wetting and Drying (AWD)":                           (0.55,  -2000,  0.05),
    "System of Rice Intensification (SRI) Water Management":        (0.60,      0,  0.30),
    "Optimised / Variable-Rate N Application":                      (0.35,   3000,  0.10),
    "Integrated Nutrient Management (INM) – Organic + Inorganic":   (0.85,      0,  0.10),
    "Manure Application":                                           (0.50,   3000,  0.12),
    "Biochar Application":                                          (2.20,  10000,  0.30),
    "Raised Bed Cultivation":                                       (1.50,  15000,  0.45),
    "Early-Maturing / Low-Emission Variety Selection":              (0.30,  -2000,  0.00),
    "High-Biomass / Deep-Root Variety Selection":                   (0.70,  -2000,  0.10),
}

practices = list(practices_data.keys())
n         = len(practices)
idx       = {p: i for i, p in enumerate(practices)}

practices_df = pd.DataFrame(
    [{"Practice": p, "Net_CSP_base": v[0],
      "Net_OC_per_ha": v[1], "Base_yield_change": v[2]}
     for p, v in practices_data.items()]
)
practices_df.to_csv(OUT_DIR / "practices.csv", index=False)
print("Saved: practices.csv")

# =============================================================================
# 2. PAIR-CATEGORY DICTIONARIES — α, β, γ independent
# =============================================================================
# Each dict maps  tuple(sorted((p1, p2)))  →  category string.
# Valid categories: "strong_pos", "pos", "neg"
# Unlisted pairs default to "zero_like" inside the sampling loop.
# =============================================================================

pair_category_alpha = {}
pair_category_beta  = {}
pair_category_gamma = {}

def set_alpha(p1, p2, cat):
    pair_category_alpha[tuple(sorted((p1, p2)))] = cat

def set_beta(p1, p2, cat):
    pair_category_beta[tuple(sorted((p1, p2)))] = cat

def set_gamma(p1, p2, cat):
    pair_category_gamma[tuple(sorted((p1, p2)))] = cat

# ── Shorthand aliases (keep code concise) ─────────────────────────────────────
CRR  = "Crop Residue Retention"
RMT  = "Reduced / Minimum Tillage"
MSD  = "Mid-Season Drainage (MSD)"
AWD  = "Alternate Wetting and Drying (AWD)"
SRI  = "System of Rice Intensification (SRI) Water Management"
VRN  = "Optimised / Variable-Rate N Application"
INM  = "Integrated Nutrient Management (INM) – Organic + Inorganic"
MA   = "Manure Application"
BA   = "Biochar Application"
RBC  = "Raised Bed Cultivation"
EMVS = "Early-Maturing / Low-Emission Variety Selection"
HBVS = "High-Biomass / Deep-Root Variety Selection"

# =============================================================================
# 2a. ALPHA — Carbon / SOC / CH4
#     pos  = co-adoption sequesters more C or suppresses more CH4 than the sum
#     neg  = co-adoption triggers priming / accelerated SOC oxidation
# =============================================================================

# ── Strong positive ───────────────────────────────────────────────────────────
for p1, p2 in [
    (CRR,  AWD),   # AWD mitigates CH4 amplification from wet residue
    (CRR,  INM),   # Double organic-C input: residue + INM amendments stack SOC
    (CRR,  BA),    # Residue-C + recalcitrant biochar-C are additive SOC pools
    (MSD,  BA),    # BA compensates drainage SOC loss; MSD suppresses CH4
    (AWD,  EMVS),  # Short submergence amplifies AWD's CH4-reduction benefit
    (SRI,  INM),   # Aerobic SRI + INM: reduced N2O and improved net SOC
    (INM,  BA),    # Biochar + organic amendments → highly stable SOC pool
]:
    set_alpha(p1, p2, "strong_pos")

# ── Moderate positive ─────────────────────────────────────────────────────────
for p1, p2 in [
    (CRR,  RMT),   # Min-till limits residue-SOC oxidation
    (CRR,  MSD),   # Drainage partially offsets residue's CH4 risk
    (CRR,  SRI),   # SRI aerobic conditions reduce residue-CH4
    (CRR,  VRN),   # Residue N reduces synthetic-N need; less N2O + SOC built
    (CRR,  MA),    # Two complementary organic-C inputs
    (CRR,  RBC),   # Raised beds reduce CH4 from residue decomposition
    (CRR,  EMVS),  # Shorter season reduces anaerobic window for residue CH4
    (CRR,  HBVS),  # Extra root biomass + surface residue contribute to SOC
    (RMT,  AWD),   # Undisturbed macropores enhance AWD drainage; both reduce C loss
    (RMT,  SRI),   # Min-till under SRI maintains structure for aerobic decomp
    (RMT,  VRN),   # Precision N on undisturbed soil reduces N2O hotspots
    (RMT,  INM),   # Organic inputs preserved in undisturbed soil → SOC accrual
    (RMT,  MA),    # Manure-C protected from rapid oxidation under min-till
    (RMT,  BA),    # Classic SOC combo; biochar stability enhanced without fragmentation
    (RMT,  HBVS),  # Deep roots in undisturbed soil → persistent rhizodeposition
    (MSD,  AWD),   # Both reduce anaerobic duration → additive CH4 suppression
    (MSD,  SRI),   # Shared CH4-suppression mechanism
    (MSD,  VRN),   # Precision N during drainage optimises N2O/CH4 trade-off
    (MSD,  RBC),   # Raised beds extend drainage efficacy
    (MSD,  EMVS),  # Short-season variety reduces anaerobic period with MSD
    (AWD,  SRI),   # Shared aerobic water-control mechanism → additive GHG benefit
    (AWD,  VRN),   # Aerobic AWD windows reduce N2O; precision N avoids over-supply
    (AWD,  INM),   # INM buffers soil biology during AWD dry cycles
    (AWD,  MA),    # Manure under aerobic cycles adds SOC without large CH4 risk
    (AWD,  RBC),   # Both facilitate drainage; raised beds extend AWD aerobic benefit
    (AWD,  HBVS),  # Deep roots contribute rhizodeposition while AWD controls CH4
    (SRI,  VRN),   # Aerobic SRI soil reduces N2O under precision N
    (SRI,  MA),    # Organic amendments under SRI aerobic conditions → SOC accrual
    (SRI,  BA),    # Biochar + SRI aerobic soil → high biochar retention
    (SRI,  RBC),   # Both improve soil aeration; positive for SOC stability
    (SRI,  EMVS),  # Aerobic soil + short anaerobic window
    (SRI,  HBVS),  # SRI supports deep-root C contribution in aerated soil
    (VRN,  INM),   # Precision mineral + organic N; N2O minimised
    (VRN,  MA),    # Organic and mineral N together reduce total N2O risk
    (VRN,  BA),    # Biochar increases N retention, making VRN more efficient
    (VRN,  RBC),   # Raised beds improve drainage and N-use efficiency
    (VRN,  EMVS),  # Short-season varieties respond well to timed N
    (VRN,  HBVS),  # High-biomass varieties uptake precision N efficiently
    (INM,  MA),    # Both add organic C; together maintain soil-biology diversity
    (INM,  RBC),   # Well-drained raised beds preserve INM-derived SOC
    (INM,  EMVS),  # Good nutrition reduces stress-driven CH4 variability
    (INM,  HBVS),  # High-biomass varieties exploit INM; deep roots maximise rhizodeposition
    (MA,   BA),    # Biochar sorbs manure-C and -N; both pools contribute to SOC
    (MA,   RBC),   # Raised beds prevent manure-C going anaerobic
    (MA,   HBVS),  # Manure supports high-biomass root growth → root-C feeds SOC
    (BA,   RBC),   # Well-drained raised beds extend biochar stability
    (BA,   EMVS),  # Biochar improves soil physical properties for short-season varieties
    (BA,   HBVS),  # Biochar + deep roots → two complementary SOC pools
    (RBC,  EMVS),  # Aerobic raised-bed suits low-emission short-season varieties
    (RBC,  HBVS),  # Raised beds allow deep roots to penetrate without waterlogging
]:
    set_alpha(p1, p2, "pos")

# ── Negative ──────────────────────────────────────────────────────────────────
for p1, p2 in [
    (AWD,  BA),    # Biochar under AWD wet-dry cycles primes native SOM
    (HBVS, MSD),   # Large root exudate flux + drainage aeration accelerates SOC oxidation
]:
    set_alpha(p1, p2, "neg")

# =============================================================================
# 2b. BETA — Cost interaction
#     pos  = cost-SAVING synergy  (dist mean will be a negative INR value)
#     neg  = extra cost BURDEN    (dist mean will be a positive INR value)
# =============================================================================

# ── Strong positive (largest cost savings) ────────────────────────────────────
for p1, p2 in [
    (CRR,  RMT),   # Shared field operations; one pass achieves both
    (MSD,  AWD),   # Shared irrigation infrastructure amortised across both
    (MSD,  VRN),   # Drainage + precision-N at same field entry; strong overlap
    (AWD,  EMVS),  # Short season reduces AWD irrigation cycles; savings compound
]:
    set_beta(p1, p2, "strong_pos")

# ── Moderate positive (cost-saving) ──────────────────────────────────────────
for p1, p2 in [
    (CRR,  MSD),   # Surface residue eases drainage-channel maintenance
    (CRR,  AWD),   # Residue reduces evaporation, lowering AWD irrigation frequency
    (CRR,  VRN),   # Residue N availability reduces total synthetic-N requirement
    (CRR,  EMVS),  # Shorter season + residue: fewer field entries required
    (RMT,  AWD),   # Min-till preserves bund integrity, cutting AWD water-gate maintenance
    (RMT,  SRI),   # Both reduce machinery passes and water use; operational overlap
    (RMT,  VRN),   # Min-till improves N retention; less fertiliser needed
    (RMT,  EMVS),  # Fewer passes needed in a short season; min-till amplifies saving
    (MSD,  SRI),   # Shared water-management infrastructure lowers per-practice cost
    (MSD,  EMVS),  # Shorter crop duration reduces MSD cycle count
    (AWD,  SRI),   # Water management schedules overlap; single monitoring effort
    (AWD,  VRN),   # AWD dry windows are optimal N-application moments; combines entries
    (SRI,  VRN),   # Sparse SRI transplanting enables VRN with minimal extra equipment
    (SRI,  EMVS),  # Shorter crop reduces SRI water-monitoring labour
    (VRN,  INM),   # VRN precision reduces synthetic N; combined with INM organic N
    (VRN,  EMVS),  # Short season simplifies VRN split-application schedule
]:
    set_beta(p1, p2, "pos")

# ── Negative (extra cost burden) ─────────────────────────────────────────────
for p1, p2 in [
    (CRR,  BA),    # BA material + residue management labour → combined burden
    (CRR,  RBC),   # RBC earthwork + residue management on raised beds add cost
    (RMT,  BA),    # BA incorporation requires passes that negate min-till savings
    (RMT,  RBC),   # Raised-bed formation requires intensive tillage → conflict
    (MSD,  BA),    # BA procurement adds material burden with no MSD saving offset
    (MSD,  RBC),   # MSD channel work + RBC earthwork → combined civil burden
    (AWD,  BA),    # BA material cost not offset by AWD water savings
    (AWD,  RBC),   # RBC capital cost adds to AWD monitoring/labour cost
    (SRI,  BA),    # SRI training/labour + BA material → two high-effort costs stack
    (SRI,  RBC),   # SRI transplanting + RBC earthwork → two expensive interventions
    (VRN,  BA),    # VRN equipment + BA material → combined capital burden
    (VRN,  RBC),   # RBC earthwork not offset by VRN savings
    (INM,  BA),    # INM cost-neutral but BA adds net material burden on top
    (INM,  RBC),   # INM organic inputs + RBC earthwork → two material-cost streams
    (MA,   BA),    # Both require material procurement; no shared operation
    (MA,   RBC),   # Manure spreading on raised beds requires extra labour
    (BA,   RBC),   # Strongest combined burden: BA material + RBC earthwork
    (BA,   EMVS),  # BA material + certified seed cost; no operational overlap
    (BA,   HBVS),  # BA adds 10 000 INR; HBVS saves only 2 000 → net burden
    (RBC,  EMVS),  # RBC earthwork + certified seed cost; no shared saving
    (RBC,  HBVS),  # RBC high earthwork cost not offset by HBVS seed savings
]:
    set_beta(p1, p2, "neg")

# =============================================================================
# 2c. GAMMA — Yield interaction
#     pos  = co-adoption amplifies yield beyond individual effects
#     neg  = co-adoption compounds yield loss / restricts individual yield gain
# =============================================================================

# ── Strong positive ───────────────────────────────────────────────────────────
for p1, p2 in [
    (SRI,  INM),   # SRI sparse planting + INM nutrition → multiplicative yield gain
    (SRI,  RBC),   # Raised beds under SRI → near-ideal root zone; strongly amplified
    (VRN,  RBC),   # Raised beds maximise N-uptake efficiency; strong yield response
    (INM,  BA),    # Biochar retains INM nutrients; soil biology activated → strong yield
    (INM,  RBC),   # Well-drained raised beds keep INM inputs fully available to roots
    (INM,  HBVS),  # HBVS has large nutrient demand; INM fully meets it
    (BA,   RBC),   # Biochar in aerated raised beds → highest plant-available W and N
    (RBC,  HBVS),  # Raised beds allow full deep-root architecture without waterlogging
]:
    set_gamma(p1, p2, "strong_pos")

# ── Moderate positive ─────────────────────────────────────────────────────────
for p1, p2 in [
    (CRR,  SRI),   # Residue improves soil-water retention, supporting SRI aerobic regime
    (CRR,  VRN),   # Residue N reduces synchrony gap in precision N supply
    (CRR,  INM),   # Combined organic matter improves nutrient supply → yield
    (CRR,  MA),    # Double organic input improves nutrient supply
    (CRR,  BA),    # Residue + biochar both improve soil physical properties
    (CRR,  RBC),   # Residue on raised beds improves water retention in bed profile
    (CRR,  HBVS),  # Deep roots access residue-derived nutrients
    (RMT,  SRI),   # Min-till preserves structure that SRI exploits for root development
    (RMT,  VRN),   # Undisturbed N-placement zones improve precision N response
    (RMT,  INM),   # Min-till prevents disruption of INM-built soil microbial networks
    (RMT,  MA),    # Manure-C protected; nutrient availability improves yield
    (RMT,  BA),    # Biochar intact in undisturbed soil → persistent yield improvement
    (RMT,  RBC),   # Raised beds maintain structure that min-till creates
    (RMT,  HBVS),  # Deep roots develop fully in undisturbed profile
    (MSD,  INM),   # INM compensates any yield penalty from drainage
    (MSD,  BA),    # BA compensates yield loss from drainage-induced nutrient leaching
    (MSD,  RBC),   # Raised beds facilitate drainage without water stress
    (AWD,  SRI),   # Well-aerated root zone from both → better yield
    (AWD,  VRN),   # Precision N applied in AWD aerobic windows → better uptake
    (AWD,  INM),   # INM buffers nutrient stress during AWD dry cycles
    (AWD,  MA),    # Manure provides organic N buffer during AWD stress
    (AWD,  BA),    # Biochar improves water-holding, buffering AWD intermittent stress
    (AWD,  RBC),   # Both create optimal root-zone aeration
    (AWD,  EMVS),  # Short-season varieties adapted to intermittent water
    (SRI,  VRN),   # Precision N under SRI sparse planting → maximised per-plant uptake
    (SRI,  MA),    # Organic N from manure supports SRI's high per-plant yield
    (SRI,  BA),    # Biochar improves root-zone water retention under SRI aerobic regime
    (SRI,  EMVS),  # Early maturity suits aerobic water-management schedule
    (SRI,  HBVS),  # SRI aerated soil allows deep roots to develop fully
    (VRN,  INM),   # Organic + precision mineral N → optimised supply without luxury loss
    (VRN,  MA),    # Organic and mineral N both at optimal timing
    (VRN,  BA),    # Biochar slow-releases VRN N → sustained yield
    (VRN,  HBVS),  # High-biomass varieties respond strongly to precision N
    (INM,  MA),    # Two organic-C streams improve nutrient supply
    (INM,  EMVS),  # Good nutrition ensures short-season varieties reach full potential
    (MA,   BA),    # Biochar sorbs manure nutrients and releases slowly
    (MA,   RBC),   # Manure nutrients fully available in well-aerated raised beds
    (MA,   HBVS),  # Manure organic N supports HBVS's large nutrient demand
    (BA,   EMVS),  # Biochar buffers soil W/N for short-season varieties
    (BA,   HBVS),  # Biochar + deep roots improve nutrient and water uptake
    (RBC,  EMVS),  # Raised beds improve aeration for short-season varieties
]:
    set_gamma(p1, p2, "pos")

# ── Negative ──────────────────────────────────────────────────────────────────
for p1, p2 in [
    (CRR,  MSD),   # Residue under drainage causes N immobilisation; compounds yield penalty
    (RMT,  MSD),   # compaction restricts drainage → increases loss
    (MSD,  AWD),   # Dual water-stress regime compounds crop water stress
    (MSD,  HBVS),  # High-biomass varieties need sustained water; drainage restricts it
    (AWD,  HBVS),  # AWD dry cycles create water stress that large canopy amplifies
]:
    set_gamma(p1, p2, "neg")

# =============================================================================
# 3. SAMPLING DISTRIBUTIONS
# =============================================================================
dist_params = {
    "Alpha": {
        "strong_pos": {"mean":  0.12, "sd": 0.03},
        "pos":        {"mean":  0.08, "sd": 0.03},
        "neg":        {"mean": -0.07, "sd": 0.02},
        "zero_like":  {"mean":  0.00, "sd": 0.015},
    },
    "Beta": {
        "strong_pos": {"mean": -1200.0, "sd":  250.0},   # large saving
        "pos":        {"mean":  -500.0, "sd":  150.0},   # moderate saving
        "neg":        {"mean":  800.0, "sd":  200.0},   # cost burden
        "zero_like":  {"mean":    0.0, "sd":  100.0},
    },
    "Gamma": {
        "strong_pos": {"mean":  0.10, "sd": 0.03},
        "pos":        {"mean":  0.05, "sd": 0.02},
        "neg":        {"mean": -0.03, "sd": 0.015},
        "zero_like":  {"mean":  0.00, "sd": 0.01},
    },
}

ALPHA_CLIP = (-0.30,  0.30)
BETA_CLIP  = (-500.0, 300.0)
GAMMA_CLIP = (-0.15,  0.15)

# =============================================================================
# 4. FILL MATRICES — α, β, γ sampled independently per dimension
# =============================================================================
SEED = 42
rng  = np.random.default_rng(SEED)

Alpha = np.zeros((n, n), dtype=float)
Beta  = np.zeros((n, n), dtype=float)
Gamma = np.zeros((n, n), dtype=float)
Delta = np.zeros((n, n), dtype=int)

for p1, p2 in combinations(practices, 2):
    key = tuple(sorted((p1, p2)))
    i, j = idx[p1], idx[p2]

    cat_a = pair_category_alpha.get(key, "zero_like")
    cat_b = pair_category_beta.get(key,  "zero_like")
    cat_g = pair_category_gamma.get(key, "zero_like")

    a = float(np.clip(rng.normal(dist_params["Alpha"][cat_a]["mean"],
                                 dist_params["Alpha"][cat_a]["sd"]),  *ALPHA_CLIP))
    b = float(np.clip(rng.normal(dist_params["Beta"][cat_b]["mean"],
                                 dist_params["Beta"][cat_b]["sd"]),   *BETA_CLIP))
    g = float(np.clip(rng.normal(dist_params["Gamma"][cat_g]["mean"],
                                 dist_params["Gamma"][cat_g]["sd"]),  *GAMMA_CLIP))

    Alpha[i, j] = Alpha[j, i] = a
    Beta[i, j]  = Beta[j, i]  = b
    Gamma[i, j] = Gamma[j, i] = g

np.fill_diagonal(Alpha, 0.0)
np.fill_diagonal(Beta,  0.0)
np.fill_diagonal(Gamma, 0.0)
np.fill_diagonal(Delta, 0)

# =============================================================================
# 5. SANITY CHECKS
# =============================================================================
assert np.allclose(Alpha, Alpha.T),        "Alpha not symmetric"
assert np.allclose(Beta,  Beta.T),         "Beta not symmetric"
assert np.allclose(Gamma, Gamma.T),        "Gamma not symmetric"
assert np.array_equal(Delta, Delta.T),     "Delta not symmetric"
print("Matrix symmetry checks passed.")

# =============================================================================
# 6. SAVE MATRICES
# =============================================================================
pd.DataFrame(Alpha, index=practices, columns=practices).to_csv(OUT_DIR / "alpha_carbon.csv")
pd.DataFrame(Beta,  index=practices, columns=practices).to_csv(OUT_DIR / "beta_cost.csv")
pd.DataFrame(Gamma, index=practices, columns=practices).to_csv(OUT_DIR / "gamma_yield.csv")
pd.DataFrame(Delta, index=practices, columns=practices).to_csv(OUT_DIR / "delta_incompatibility.csv")
print("Saved: alpha_carbon.csv, beta_cost.csv, gamma_yield.csv, delta_incompatibility.csv")

# =============================================================================
# 7. QUICK SUMMARY
# =============================================================================
def _count(d, cat): return sum(1 for v in d.values() if v == cat)

print("\n── Data generation complete ──────────────────────────────────────────")
print(f"  Practices            : {n}")
print(f"  Total pairs          : {n * (n - 1) // 2}")
print()
print("  Alpha (Carbon)")
print(f"    strong_pos         : {_count(pair_category_alpha, 'strong_pos')}")
print(f"    pos                : {_count(pair_category_alpha, 'pos')}")
print(f"    neg                : {_count(pair_category_alpha, 'neg')}")
print(f"    zero_like (default): {n*(n-1)//2 - len(pair_category_alpha)}")
print()
print("  Beta (Cost)")
print(f"    strong_pos (saving): {_count(pair_category_beta, 'strong_pos')}")
print(f"    pos (saving)       : {_count(pair_category_beta, 'pos')}")
print(f"    neg (burden)       : {_count(pair_category_beta, 'neg')}")
print(f"    zero_like (default): {n*(n-1)//2 - len(pair_category_beta)}")
print()
print("  Gamma (Yield)")
print(f"    strong_pos         : {_count(pair_category_gamma, 'strong_pos')}")
print(f"    pos                : {_count(pair_category_gamma, 'pos')}")
print(f"    neg                : {_count(pair_category_gamma, 'neg')}")
print(f"    zero_like (default): {n*(n-1)//2 - len(pair_category_gamma)}")