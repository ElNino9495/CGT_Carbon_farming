# Cooperative Carbon Farming Analysis Framework

This repository contains a Cooperative Game Theory (CGT) framework for optimizing and analyzing carbon credit coalitions among smallholder farmers. It models the adoption of sustainable agricultural practices, minimizes costs / maximizes carbon revenue, and determines stable surplus allocations using cooperative game solution concepts.

---

## Directory Structure

```
├── data/
│   ├── Dataset_INR/                   # Raw input CSVs (farmers, practices, interaction matrices)
│   ├── processed/                     # Output directory — one subfolder per experiment
│   ├── farmer_generator.py            # Generates synthetic farmer profiles
│   └── data_generator.py              # Generates practice interaction matrices
└── notebooks/
    ├── config.py                      # Central configuration file
    ├── launch_experiments.py          # Master script for launching batch experiments
    ├── run_experiment.py              # Script to run a single experiment end-to-end
    ├── notebook_01_data_ingestion.py  # Data extraction & validation
    ├── notebook_02_solo_optimisation.py # Phase 1: Solo optimization
    ├── notebook_03_coalition_enumeration.py # Phase 1: Coalition formation
    ├── notebook_04_surplus_allocation.py # Phase 2: Allocation mechanisms
    └── ...                            # Original Jupyter notebooks (.ipynb)
```

---

## Running Experiments

This framework supports both **automated batch processing** (for ablation studies) and **manual interactive execution** (for debugging).

### 1. Automated Batch Experiments (Recommended)

Use `launch_experiments.py` to run multiple experiments in parallel. This is ideal for sensitivity analysis (e.g., varying Carbon Credit Price or MRV costs).

**Step 1:** Open `notebooks/launch_experiments.py` and modify the `experiments` list. Each tuple defines a unique configuration:

```python
experiments = [
    # (name, cf_rule, ccp, delta_mrv, fixed_mrv, var_mrv, fixed_t, var_t, budget, paddy, n_jobs)
    ("CF1_delta00", "CF1", 1500, 0.0, 5000, 4000, 2000, 500, 1.0, 22000, 4),
    ...
]
```

**Step 2:** Run the launcher from the root directory:

```bash
python notebooks/launch_experiments.py
```

This script manages parallel execution using `ProcessPoolExecutor`, running multiple experiments concurrently based on available CPU cores.

### 2. Single Experiment via CLI

Use `run_experiment.py` to run a specific configuration without editing files. This is useful for testing a single scenario.

```bash
python notebooks/run_experiment.py \
  --name "My_Test_Run" \
  --cf_rule "CF1" \
  --ccp 2000 \
  --delta_mrv 0.7 \
  --n_jobs 4
```

This script automatically executes the four pipeline steps in order (`01` -> `02` -> `03` -> `04`) using the provided parameters.

### 3. Manual / Interactive Execution

You can still run the Jupyter notebooks interactively. This is useful for debugging logic or inspecting intermediate dataframes.

1. Open `notebooks/config.py` and set your desired parameters (e.g., `EXPERIMENT_NAME`, `CF_RULE`, `CCP`).
2. Run the notebooks in sequential order:
    - `01_Data_Ingestion_and_Preprocessing.ipynb`
    - `02_Phase1_Solo_Optimisation.ipynb`
    - `03_Phase1_Characteristic_Function.ipynb`
    - `04_Phase2_Allocation.ipynb`

---

## Configuration System

The project uses a hierarchical configuration system:

1. **`notebooks/config.py`**: Defines all default values, types, and derived constants.
2. **Environment Variables**: Can override any value in `config.py` at runtime. The key is usually the parameter name (e.g., `CCP`, `DELTA_MRV`, `EXPERIMENT_NAME`).

When using `launch_experiments.py` or `run_experiment.py`, the scripts automatically handle these environment variable overrides for you.

### Key Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `EXPERIMENT_NAME` | `str` | Name of the output subfolder in `data/processed/` |
| `CF_RULE` | `str` | **Coalition Feasibility Rule**:<br>`CF1` (Pooled residual; linear), `CF2` (Area-weighted; strict), `CF3` (Sequestration-weighted; strictest) |
| `CCP` | `float` | **Carbon Credit Price** (INR / tCO₂e). |
| `DELTA_MRV` | `float` | **MRV Scale Exponent** (0.0 = const, 1.0 = linear). Values < 1 imply economies of scale. |
| `FIXED_MRV` | `float` | Flat MRV cost per certification event (INR). |
| `VARIABLE_MRV` | `float` | Variable MRV rate (INR / ha^δ). |
| `FIXED_T` | `float` | Flat transaction/registry fee (INR). |
| `VARIABLE_T` | `float` | Per-member transaction cost (INR / farmer). |
| `BUDGET_MULTIPLIER`| `float` | Scaling factor for farmer budgets (1.0 = base). |
| `PADDY_PRICE` | `float` | Farm-gate paddy price (INR / ton). |
| `N_JOBS` | `int` | Parallel workers for coalition enumeration (-1 = all cores). |

---

## Outputs

All results are saved in `data/processed/<EXPERIMENT_NAME>/`.

- **`optimization_inputs.pkl`**: Frozen copy of input data.
- **`standalone_values.pkl`**: Results from Solo Optimization (Phase 1).
- **`characteristic_function.pkl`**: Results from Coalition Enumeration (Phase 1).
- **`allocation_results.pkl`**: Final allocations from Phase 2.
- **`plots/`**: Stability analysis and allocation plots (generated by Phase 2).

---

## Prerequisites

- Python 3.9+
- Gurobi Optimizer (valid licence required — academic licences available free at [gurobi.com](https://www.gurobi.com))
- Key Dependencies:
  ```bash
  pip install gurobipy pandas numpy matplotlib joblib tqdm
  ```
