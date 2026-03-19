# =============================================================================
# config.py — Central parameter file for the carbon farming coalition pipeline
# =============================================================================
import os
from pathlib import Path



# Find the project root (assuming this file is in 'notebooks/')
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR_PATH = PROJECT_ROOT / "data" / "Dataset_INR"
PROCESSED_DIR_PATH = PROJECT_ROOT / "data" / "processed"

# Convert to string to avoid PosixPath issues in some pandas versions
DATA_DIR       = str(DATA_DIR_PATH)
PROCESSED_DIR  = str(PROCESSED_DIR_PATH)

FARMERS_15_CSV = f"{DATA_DIR}/15_farmers.csv"
FARMERS_25_CSV = f"{DATA_DIR}/25_farmers.csv"
PRACTICES_CSV  = f"{DATA_DIR}/practices.csv"
ALPHA_CSV      = f"{DATA_DIR}/alpha_carbon.csv"
BETA_CSV       = f"{DATA_DIR}/beta_cost.csv"
GAMMA_CSV      = f"{DATA_DIR}/gamma_yield.csv"
DELTA_CSV      = f"{DATA_DIR}/delta_incompatibility.csv"

INPUTS_PKL             = f"{PROCESSED_DIR}/optimization_inputs.pkl"
STANDALONE_PKL         = f"{PROCESSED_DIR}/standalone_values.pkl"
CHARACTERISTIC_FN_PKL  = f"{PROCESSED_DIR}/characteristic_function.pkl"

# ── Carbon market ─────────────────────────────────────────────────────────────
CCP = 1500          # INR / tCO2e  — carbon credit price

# ── Farm economics ────────────────────────────────────────────────────────────
PADDY_PRICE = 22_000    # INR / ton — farm-gate paddy price

# ── MRV (Measurement, Reporting & Verification) costs ────────────────────────
FIXED_MRV = 5_000       # INR — flat cost regardless of coalition size or area.
VARIABLE_MRV = 2_000    # INR / ha^DELTA_MRV — scales with total certified area.
DELTA_MRV = 0.7         # Exponent on total area in the MRV cost formula.

# ── Transaction / registry costs ─────────────────────────────────────────────
FIXED_T = 2_000         # INR — flat registry/platform fee per certification.
VARIABLE_T = 500        # INR / farmer — per-member administrative cost.

# ── Solver settings ───────────────────────────────────────────────────────────
GUROBI_TIME_LIMIT  = 60       # seconds per solve (increase for harder instances)
GUROBI_MIP_GAP     = 1e-4     # relative MIP optimality gap
GUROBI_OUTPUT_FLAG = 0        # 0 = silent, 1 = verbose

# ── Coalition enumeration ─────────────────────────────────────────────────────
N_JOBS = -1     # joblib parallel workers; -1 = use all available CPU cores

# ── Numerical tolerances ──────────────────────────────────────────────────────
SYMMETRY_TOL   = 1e-8   # max allowed |A_ij - A_ji| for matrix symmetry check
EPSILON_MAX    = 1e-4   # least-core ε* below this → treat coalition as stable

# ── Derived / convenience constants ──────────────────────────────────────────
def mrv_cost(total_ha: float) -> float:
    """Total MRV cost for a coalition with `total_ha` hectares combined."""
    return FIXED_MRV + VARIABLE_MRV * (total_ha ** DELTA_MRV)

def transaction_cost(n_farmers: int) -> float:
    """Total transaction cost for a coalition of `n_farmers` members."""
    return FIXED_T + VARIABLE_T * n_farmers

def certification_cost(total_ha: float, n_farmers: int) -> float:
    """Combined MRV + transaction cost for a coalition."""
    return mrv_cost(total_ha) + transaction_cost(n_farmers)