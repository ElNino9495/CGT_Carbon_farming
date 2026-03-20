

#!/usr/bin/env python
# coding: utf-8

# # Notebook 01 — Data Ingestion and Preprocessing
# 
# **Purpose:** Load all raw CSVs, validate them, and save a single clean pickle
# (`optimization_inputs.pkl`) that every downstream notebook reads from.
# 
# 
# 
# ---
# **Outputs:**
# - `data/processed/optimization_inputs.pkl` — dict with keys:
#   `farmers_15`, `farmers_25`, `practices`, `Alpha`, `Beta`, `Gamma`, `Delta`,
#   `practice_names`, `practice_index`, `config`

# ## 0. Imports and setup

# In[1]:


import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure config.py is importable regardless of working directory
sys.path.insert(0, str(Path.cwd()))
import config
try:
    display
except NameError:
    def display(x): print(x.to_string() if hasattr(x, "to_string") else str(x))

# Create processed output directory if it doesn't exist
Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

print("Python path OK")
print(f"Data directory   : {config.DATA_DIR}")
print(f"Processed dir    : {config.PROCESSED_DIR}")



# ## 1. Load farmer data

# In[2]:


farmers_15 = pd.read_csv(config.FARMERS_15_CSV)
farmers_25 = pd.read_csv(config.FARMERS_25_CSV)

print(f"15-farmer dataset shape : {farmers_15.shape}")
print(f"25-farmer dataset shape : {farmers_25.shape}")
print()
print("Columns:", list(farmers_15.columns))
print()
farmers_15.head()


# In[3]:


# Quick descriptive statistics
print("=== 15 farmers ===")
display(farmers_15[["Farm_Size_ha", "Budget_total_INR_per_season"]].describe().round(2))

print("\n=== 25 farmers ===")
display(farmers_25[["Farm_Size_ha", "Budget_total_INR_per_season"]].describe().round(2))


# ## 2. Load practice data

# In[4]:


practices = pd.read_csv(config.PRACTICES_CSV)

print(f"Number of practices : {len(practices)}")
print(f"Columns             : {list(practices.columns)}")
print()
practices


# In[5]:


# Build ordered name list and name→index mapping (used by all notebooks)
practice_names = practices["Practice"].tolist()
practice_index = {name: i for i, name in enumerate(practice_names)}

M = len(practice_names)  # number of practices
print(f"M = {M} practices")
for i, p in enumerate(practice_names):
    print(f"  [{i:2d}] {p}")


# ## 3. Load interaction matrices

# In[6]:


def load_matrix(csv_path: str, practice_names: list, dtype=float) -> np.ndarray:
    """
    Load a 20×20 interaction matrix CSV.
    The CSV has practice names as both the index column and column headers.
    Returns a numpy array ordered according to `practice_names`.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Reindex rows and columns to match practice_names ordering
    df = df.reindex(index=practice_names, columns=practice_names)
    return df.values.astype(dtype)

Alpha = load_matrix(config.ALPHA_CSV, practice_names, dtype=float)
Beta  = load_matrix(config.BETA_CSV,  practice_names, dtype=float)
Gamma = load_matrix(config.GAMMA_CSV, practice_names, dtype=float)
Delta = load_matrix(config.DELTA_CSV, practice_names, dtype=int)

print(f"Alpha shape : {Alpha.shape}   dtype: {Alpha.dtype}")
print(f"Beta  shape : {Beta.shape}    dtype: {Beta.dtype}")
print(f"Gamma shape : {Gamma.shape}   dtype: {Gamma.dtype}")
print(f"Delta shape : {Delta.shape}   dtype: {Delta.dtype}")


# ## 4. Validation

# In[7]:


# ── 4a. NaN checks ────────────────────────────────────────────────────────────
print("NaN checks")
print("  farmers_15   :", farmers_15.isnull().sum().sum(), "NaNs")
print("  farmers_25   :", farmers_25.isnull().sum().sum(), "NaNs")
print("  practices    :", practices.isnull().sum().sum(),  "NaNs")
print("  Alpha        :", np.isnan(Alpha).sum(),           "NaNs")
print("  Beta         :", np.isnan(Beta).sum(),            "NaNs")
print("  Gamma        :", np.isnan(Gamma).sum(),           "NaNs")
print("  Delta        :", np.isnan(Delta.astype(float)).sum(), "NaNs")


# In[8]:


# ── 4b. Symmetry checks ───────────────────────────────────────────────────────
tol = config.SYMMETRY_TOL
checks = {
    "Alpha" : np.allclose(Alpha, Alpha.T, atol=tol),
    "Beta"  : np.allclose(Beta,  Beta.T,  atol=tol),
    "Gamma" : np.allclose(Gamma, Gamma.T, atol=tol),
    "Delta" : np.array_equal(Delta, Delta.T),
}
print("Symmetry checks (tolerance =", tol, ")")
all_ok = True
for name, ok in checks.items():
    status = "PASS" if ok else "FAIL"
    print(f"  {name:6s} : {status}")
    if not ok:
        all_ok = False

assert all_ok, "One or more matrices failed the symmetry check — inspect the CSVs."


# In[9]:


# ── 4c. Zero-diagonal checks ──────────────────────────────────────────────────
print("Zero-diagonal checks")
for name, mat in [("Alpha", Alpha), ("Beta", Beta), ("Gamma", Gamma), ("Delta", Delta)]:
    diag_sum = np.abs(np.diag(mat)).sum()
    status = "PASS" if diag_sum == 0 else f"FAIL (sum={diag_sum})"
    print(f"  {name:6s} diagonal : {status}")


# In[10]:


# ── 4d. Delta value check — must be binary (0 or 1) ──────────────────────────
unique_delta = np.unique(Delta)
assert set(unique_delta).issubset({0, 1}), f"Delta contains non-binary values: {unique_delta}"
print(f"Delta values : {unique_delta}  — binary check PASS")


# In[11]:


# ── 4e. Budget sanity — every budget must be > 0 ──────────────────────────────
for label, df in [("15 farmers", farmers_15), ("25 farmers", farmers_25)]:
    neg = (df["Budget_total_INR_per_season"] <= 0).sum()
    assert neg == 0, f"{label}: {neg} farmers have non-positive budget"
    print(f"Budget check ({label}): all positive — PASS")


# ## 5. Incompatible practice pairs

# In[12]:


# Collect upper-triangle pairs where Delta_jk = 1
incompatible_pairs = []
for j in range(M):
    for k in range(j + 1, M):
        if Delta[j, k] == 1:
            incompatible_pairs.append((j, k, practice_names[j], practice_names[k]))

print(f"Total incompatible pairs: {len(incompatible_pairs)}")
print()
print("-" * 80)
for idx_j, idx_k, p1, p2 in incompatible_pairs:
    print(f"  [{idx_j:2d}] {p1}")
    print(f"  [{idx_k:2d}] {p2}")
    print(f"  → These two practices cannot be co-adopted (Δ = 1)")
    print("-" * 80)


# ## 6. Summary statistics on interaction matrices

# In[13]:


def upper_triangle_stats(mat: np.ndarray, name: str, unit: str) -> None:
    """
    Print statistics on the upper triangle (j < k) of a symmetric matrix.
    Incompatible pairs (Delta_jk=1) are excluded because their interaction
    values are structurally zero and irrelevant to the optimiser.
    """
    vals = []
    for j in range(M):
        for k in range(j + 1, M):
            if Delta[j, k] == 0:   # skip incompatible pairs
                vals.append(mat[j, k])
    vals = np.array(vals)
    nonzero = (vals != 0).sum()
    print(f"{name} ({unit}) — compatible pairs only")
    print(f"  n pairs  : {len(vals)}")
    print(f"  nonzero  : {nonzero}")
    print(f"  min      : {vals.min():.4f}")
    print(f"  max      : {vals.max():.4f}")
    print(f"  mean     : {vals.mean():.4f}")
    print(f"  positive : {(vals > 0).sum()}   negative: {(vals < 0).sum()}")
    print()

upper_triangle_stats(Alpha, "Alpha", "tCO2e/ha/season")
upper_triangle_stats(Beta,  "Beta",  "INR/ha/season")
upper_triangle_stats(Gamma, "Gamma", "tons/ha/season")


# ## 7. Config parameter echo

# In[14]:


print("Economic parameters loaded from config.py")
print("-" * 50)
print(f"  CCP (carbon credit price)   : {config.CCP:>10,} INR/tCO2e")
print(f"  PADDY_PRICE                 : {config.PADDY_PRICE:>10,} INR/ton")
print(f"  FIXED_MRV                   : {config.FIXED_MRV:>10,} INR")
print(f"  VARIABLE_MRV                : {config.VARIABLE_MRV:>10,} INR/ha^delta")
print(f"  DELTA_MRV                   : {config.DELTA_MRV:>10}")
print(f"  FIXED_T                     : {config.FIXED_T:>10,} INR")
print(f"  VARIABLE_T                  : {config.VARIABLE_T:>10,} INR/farmer")
print("-" * 50)

# Quick sanity: what does solo certification cost for a 1-ha farmer?
solo_mrv = config.mrv_cost(1.0)
solo_t   = config.transaction_cost(1)
print(f"\nSolo certification cost (1 ha farmer):")
print(f"  MRV cost   : {solo_mrv:,.0f} INR")
print(f"  Trans cost : {solo_t:,.0f} INR")
print(f"  Total      : {solo_mrv + solo_t:,.0f} INR")

# And for a 10-ha coalition of 5 farmers?
coal_mrv = config.mrv_cost(10.0)
coal_t   = config.transaction_cost(5)
print(f"\nCoalition certification cost (5 farmers, 10 ha total):")
print(f"  MRV cost   : {coal_mrv:,.0f} INR")
print(f"  Trans cost : {coal_t:,.0f} INR")
print(f"  Total      : {coal_mrv + coal_t:,.0f} INR")


# ## 8. Save processed inputs pickle

# In[15]:


# Build the config snapshot dict (only serialisable values — no functions)
config_snapshot = {
    "CCP"          : config.CCP,
    "PADDY_PRICE"  : config.PADDY_PRICE,
    "FIXED_MRV"    : config.FIXED_MRV,
    "VARIABLE_MRV" : config.VARIABLE_MRV,
    "DELTA_MRV"    : config.DELTA_MRV,
    "FIXED_T"      : config.FIXED_T,
    "VARIABLE_T"   : config.VARIABLE_T,
    "SYMMETRY_TOL" : config.SYMMETRY_TOL,
    "EPSILON_MAX"  : config.EPSILON_MAX,
}

optimization_inputs = {
    # Farmer DataFrames
    "farmers_15"      : farmers_15,
    "farmers_25"      : farmers_25,

    # Practice DataFrame
    "practices"       : practices,

    # Practice naming / indexing
    "practice_names"  : practice_names,   # list[str], length M
    "practice_index"  : practice_index,   # dict str→int

    # Interaction matrices (all M×M numpy arrays)
    "Alpha"           : Alpha,   # tCO2e/ha/season — sequestration synergies
    "Beta"            : Beta,    # INR/ha/season   — cost interactions
    "Gamma"           : Gamma,   # tons/ha/season  — yield interactions
    "Delta"           : Delta,   # binary          — incompatibility (0/1)

    # Incompatible pair list for fast lookup [(j, k, name_j, name_k), ...]
    "incompatible_pairs" : incompatible_pairs,

    # Economic parameters
    "config"          : config_snapshot,
}

out_path = Path(config.INPUTS_PKL)
with open(out_path, "wb") as f:
    pickle.dump(optimization_inputs, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved: {out_path.resolve()}")
print(f"Keys  : {list(optimization_inputs.keys())}")


# ## 9. Reload and verify the pickle

# In[16]:


with open(config.INPUTS_PKL, "rb") as f:
    loaded = pickle.load(f)

# Structural checks on the reloaded object
assert set(loaded.keys()) == set(optimization_inputs.keys()), "Key mismatch after reload"
assert np.array_equal(loaded["Alpha"], Alpha), "Alpha mismatch after reload"
assert np.array_equal(loaded["Delta"], Delta), "Delta mismatch after reload"
assert len(loaded["practice_names"]) == M,     "practice_names length mismatch"
assert len(loaded["farmers_15"]) == 15,         "farmers_15 row count mismatch"
assert len(loaded["farmers_25"]) == 25,         "farmers_25 row count mismatch"

print("Pickle reload checks: all PASS")
print(f"\nFile size: {out_path.stat().st_size / 1024:.1f} KB")


# ## 10. Final summary

# In[17]:


print("=" * 60)
print("NOTEBOOK 01 COMPLETE")
print("=" * 60)
print(f"  Practices loaded         : {M}")
print(f"  Incompatible pairs       : {len(incompatible_pairs)}")
print(f"  Farmers (15-set)         : {len(farmers_15)}")
print(f"  Farmers (25-set)         : {len(farmers_25)}")
print(f"  Matrices validated       : Alpha, Beta, Gamma, Delta")
print(f"  Output pickle            : {config.INPUTS_PKL}")
print()


# In[ ]:




