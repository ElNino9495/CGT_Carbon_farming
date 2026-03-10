from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class ModelConfig:
    # Economic parameters
    CCP: float = 1500.0              # INR per tCO2e (carbon credit price)
    CROP_PRICE: float = 22000.0      # INR per ton of crop

    # Certification cost parameters
    fixed_MR: float = 50000.0        # Fixed monitoring/reporting cost (INR)
    variable_MR: float = 500.0       # Variable MR cost
    alpha_MR: float = 0.7            # Economies of scale exponent for MR
    phi_MR: float = 5000.0           # Per-practice MR overhead (INR)

    fixed_V: float = 80000.0         # Fixed verification cost (INR)
    variable_V: float = 300.0        # Variable verification cost
    beta_V: float = 0.6              # Economies of scale exponent for verification
    phi_V: float = 8000.0            # Per-practice verification overhead (INR)

    fixed_T: float = 10000.0         # Fixed transaction/registry cost (INR)
    variable_T: float = 500.0        # Per-farmer transaction cost (INR)

    # Solver settings
    solver: str = "pulp"           # "gurobi", "pulp", "cplex"
    time_limit: int = 120            # seconds per solve
    mip_gap: float = 1e-4

    # GRASP heuristic settings (fallback / warm-start)
    grasp_iters: int = 300
    grasp_alpha: float = 0.3
    grasp_seeds: list = field(default_factory=lambda: [1, 7, 11, 21, 42])