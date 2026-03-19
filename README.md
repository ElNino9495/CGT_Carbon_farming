# Cooperative Carbon Farming Analysis Framework

This repository contains a Cooperative Game Theory (CGT) framework for optimizing and analyzing carbon credit coalitions among smallholder farmers. It models the adoption of sustainable agricultural practices, minimizes costs / maximizes carbon revenue, and determines stable surplus allocations using cooperative game solution concepts.

---

## Repository Structure

```
├── config.py                          # Central configuration file
├── data/
│   ├── Dataset_INR/                   # Raw input CSVs (farmers, practices, interaction matrices)
│   ├── processed/                     # Output directory — one subfolder per experiment
│   ├── farmer_generator.py            # Generates synthetic farmer profiles (budgets, sizes)
│   └── data_generator.py              # Generates practice interaction matrices (α, β, γ)
└── notebooks/
    ├── 01_Data_Ingestion_and_Preprocessing.ipynb
    ├── 02_Phase1_Solo_Optimisation.ipynb
    ├── 03_Phase1_Characteristic_Function.ipynb
    └── 04_Phase2_Allocation.ipynb
```

| Notebook | Purpose |
|----------|---------|
| `01` | Ingests raw CSVs and creates a frozen input snapshot |
| `02` | Solves the individual MILP for each farmer (Disagreement Point) |
| `03` | Enumerates all $2^N$ coalitions and computes $v(S)$ |
| `04` | Computes Core, Shapley, Nucleolus allocations and stability plots |

---

## How to Run an Experiment (Ablation Study)

To run a new experiment (e.g., testing "High Budget" vs "Low Budget" or comparing CF1 vs CF2 rules), follow this workflow. Each experiment is fully isolated and reproducible.

### Step 1 — Configure the run

Open `config.py` and edit the **Experiment Configuration** section:

```python
# 1. Name your experiment — this creates a unique folder for all results.
EXPERIMENT_NAME = "HighBudget_CF2_Run"

# 2. Select the Coalition Feasibility Rule.
CF_RULE = "CF2"   # "CF1", "CF2", or "CF3"
```

### Step 2 — Generate or modify input data (optional)

Only needed if you are changing the "physics" of the problem — e.g., farmer budgets, farm sizes, or global prices.

1. Open `data/farmer_generator.py` and edit the parameters in `main()`:

```python
# Example: doubling the per-hectare budget
budget_per_ha = 7000.0
```

2. Run the generator from a terminal:

```bash
python data/farmer_generator.py
```

> **Note:** This overwrites the CSVs in `Dataset_INR/`. This is safe because Step 3 will save a frozen snapshot of the data into your experiment folder before anything is modified.

### Step 3 — Execute the pipeline

Run the four notebooks **in numerical order**. Each notebook reads `EXPERIMENT_NAME` from `config.py` and automatically reads from / writes to the correct experiment subfolder under `data/processed/`.

**`01_Data_Ingestion_and_Preprocessing.ipynb`**
- Reads raw CSVs from `Dataset_INR/`
- Validates matrices (symmetry, zero diagonal, no NaNs)
- Saves `optimization_inputs.pkl` to `data/processed/HighBudget_CF2_Run/`
- ✅ Your inputs are now frozen — future CSV changes won't affect this experiment.

**`02_Phase1_Solo_Optimisation.ipynb`**
- Solves the individual MILP for each farmer using the frozen inputs
- Computes each farmer's standalone value $\tilde{v}(\{i\})$ (IR floor)
- Saves `standalone_values.pkl` to the experiment folder

**`03_Phase1_Characteristic_Function.ipynb`**
- Enumerates all $2^N$ coalitions and solves a joint MILP per coalition
- Uses the `CF_RULE` defined in Step 1
- Saves `characteristic_function.pkl`
- ⚠️ Computationally intensive for $N > 15$ — parallelised via `joblib`

**`04_Phase2_Allocation.ipynb`**
- Loads the characteristic function and computes five allocation mechanisms:
  - Least Core, Nucleolus, Shapley Value, Equal Split, Proportional Split
- Computes per-farmer carbon transfers (actual manager payouts)
- Saves `allocation_results.pkl` and generates all stability and comparison plots

---

## Key Configuration Parameters (`config.py`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `EXPERIMENT_NAME` | `str` | Name of the subfolder in `processed/` where all results are stored |
| `CF_RULE` | `str` | `"CF1"` — Pooled residual (linear; most permissive) <br> `"CF2"` — Area-weighted per-farmer share (linear; stricter) <br> `"CF3"` — Sequestration-weighted per-farmer share (non-convex; strictest) |
| `CCP` | `float` | Carbon credit price (INR / tCO₂e) |
| `PADDY_PRICE` | `float` | Farm-gate paddy price (INR / ton) |
| `FIXED_MRV` | `float` | Flat MRV cost per certification (INR) |
| `VARIABLE_MRV` | `float` | Variable MRV rate (INR / ha^δ) |
| `DELTA_MRV` | `float` | MRV scale exponent — values < 1 produce economies of scale |
| `FIXED_T` | `float` | Flat transaction / registry fee per certification (INR) |
| `VARIABLE_T` | `float` | Per-member transaction cost (INR / farmer) |
| `GUROBI_TIME_LIMIT` | `int` | Solver time limit per coalition solve (seconds) |
| `GUROBI_MIP_GAP` | `float` | Relative MIP optimality gap |
| `N_JOBS` | `int` | Parallel workers for coalition enumeration (`-1` = all CPU cores) |

---

## Prerequisites

- Python 3.9+
- Gurobi Optimizer (valid licence required — academic licences available free at [gurobi.com](https://www.gurobi.com))
- Key Python libraries:

```bash
pip install gurobipy pandas numpy matplotlib joblib tqdm
```