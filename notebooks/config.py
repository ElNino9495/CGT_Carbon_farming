# =============================================================================
# config.py — Central parameter file for the carbon farming coalition pipeline
# =============================================================================
import os
import time
from pathlib import Path

# Find the project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Paths - Inputs (Independent of experiment)
DATA_DIR_PATH = PROJECT_ROOT / "data" / "Dataset_INR"
DATA_DIR       = str(DATA_DIR_PATH)

FARMERS_15_CSV = f"{DATA_DIR}/15_farmers.csv"
FARMERS_25_CSV = f"{DATA_DIR}/25_farmers.csv"
PRACTICES_CSV  = f"{DATA_DIR}/practices.csv"
ALPHA_CSV      = f"{DATA_DIR}/alpha_carbon.csv"
BETA_CSV       = f"{DATA_DIR}/beta_cost.csv"
GAMMA_CSV      = f"{DATA_DIR}/gamma_yield.csv"
DELTA_CSV      = f"{DATA_DIR}/delta_incompatibility.csv"

# =============================================================================
#  ENV OVERRIDE HELPER
# =============================================================================
def _env(key: str, default):
    """Read an EXP_<KEY> environment variable override if present, cast to type of default."""
    raw = os.environ.get(f"EXP_{key}")
    if raw is None:
        return default
    if isinstance(default, float): return float(raw)
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            # Handle float strings like "1500.0" passed to int params
            return int(float(raw))
    return raw  # str


# =============================================================================
#  CONFIGURATION PARAMETERS
# =============================================================================

# Carbon market
CCP = _env("CCP", 1500)              # INR / tCO2e  — carbon credit price, base value 1500

# Farm economics
PADDY_PRICE = _env("PADDY_PRICE", 22_000)    # INR / ton — farm-gate paddy price

BUDGET_MULTIPLIER = _env("BUDGET_MULTIPLIER", 1.0)  # the present base value is 3.5k, so change accordingly

# FOR notebooks/notebook_03_coalition_enumeration.ipynb
CF_RULE = _env("CF_RULE", 'CF1')     # CHANGE TO CF2 or CF3 as needed

# MRV (Measurement, Reporting & Verification) costs
FIXED_MRV    = _env("FIXED_MRV",    5_000)  # INR — flat cost regardless of coalition size or area.
VARIABLE_MRV = _env("VARIABLE_MRV", 2_000)  # INR / ha^DELTA_MRV — scales with total certified area base as 2k.
DELTA_MRV    = _env("DELTA_MRV",    0.0)    # Exponent on total area in the MRV cost formula. base 0.7

# Transaction / registry costs
FIXED_T    = _env("FIXED_T",    2_000)  # INR — flat registry/platform fee per certification.
VARIABLE_T = _env("VARIABLE_T", 500)    # INR / farmer — per-member administrative cost.

# =============================================================================
#  EXPERIMENT NAMING & OUTPUT PATHS
# =============================================================================

# Automatically formatted based on active config parameters
# Can be fully overridden via EXP_EXPERIMENT_NAME
EXPERIMENT_NAME = _env(
    "EXPERIMENT_NAME",
    f"Experiment_{CF_RULE}_DELTA{DELTA_MRV}_BUDGET{BUDGET_MULTIPLIER}"
)

# Outputs go to a specific subfolder for this experiment
# e.g., data/processed/Experiment_CF1_DELTA0.0_BUDGET1.0/
PROCESSED_DIR_PATH = PROJECT_ROOT / "data" / "processed" / EXPERIMENT_NAME

# Create the folder immediately so 01_Data_Ingestion doesn't crash
PROCESSED_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Convert to string to avoid PosixPath issues in some pandas versions
PROCESSED_DIR  = str(PROCESSED_DIR_PATH)

INPUTS_PKL             = f"{PROCESSED_DIR}/optimization_inputs.pkl"
STANDALONE_PKL         = f"{PROCESSED_DIR}/standalone_values.pkl"
CHARACTERISTIC_FN_PKL  = f"{PROCESSED_DIR}/characteristic_function.pkl"

# Solver settings
GUROBI_TIME_LIMIT  = _env("GUROBI_TIME_LIMIT", 60)    # seconds per solve
GUROBI_MIP_GAP     = _env("GUROBI_MIP_GAP",    1e-4)  # relative MIP optimality gap
GUROBI_OUTPUT_FLAG = _env("GUROBI_OUTPUT_FLAG", 0)     # 0 = silent, 1 = verbose

# Coalition enumeration
N_JOBS = _env("N_JOBS", -1)  # joblib parallel workers; -1 = use all available CPU cores

# Numerical tolerances
SYMMETRY_TOL = _env("SYMMETRY_TOL", 1e-8)  # max allowed |A_ij - A_ji| for matrix symmetry check
EPSILON_MAX  = _env("EPSILON_MAX",  1e-4)  # least-core ε* below this → treat coalition as stable

# =============================================================================
#  Derived / convenience constants
# =============================================================================
def mrv_cost(total_ha: float) -> float:
    """Total MRV cost for a coalition with `total_ha` hectares combined."""
    return FIXED_MRV + VARIABLE_MRV * (total_ha ** DELTA_MRV)

def transaction_cost(n_farmers: int) -> float:
    """Total transaction cost for a coalition of `n_farmers` members."""
    return FIXED_T + VARIABLE_T * n_farmers

def certification_cost(total_ha: float, n_farmers: int) -> float:
    """Combined MRV + transaction cost for a coalition."""
    return mrv_cost(total_ha) + transaction_cost(n_farmers)