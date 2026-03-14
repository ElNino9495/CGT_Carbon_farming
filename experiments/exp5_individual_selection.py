"""
Experiment 1: Solve optimal practice selection for all farmers
using the friend-style model within this project structure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ModelConfig
from src.data_loader import load_data
from src.practice_selection import solve_all_farmers


def main():
    # Gurobi settings:
    # - use Gurobi solver
    # - objective interpreted as total INR/season
    # - budget interpreted as total budget
    config = ModelConfig(
        CCP=1500.0,         # change to 1200.0 if you want exact friend match
        CROP_PRICE=22000.0,
        solver="gurobi",    # switched from PuLP/CBC to Gurobi
        time_limit=60,
    )

    print("Loading data...")
    data = load_data("data/synthetic")
    print(f"  {len(data.farmers)} farmers, {len(data.practices)} practices")
    print(f"  {len(data.alpha)} alpha pairs, {len(data.gamma)} gamma pairs, {len(data.delta)} delta pairs")
    print(f"  {len(data.incompatible)} incompatible pairs")

    print("\nSolving optimal practice selection for each farmer...")
    results = solve_all_farmers(data, config)

    rows = []
    farmer_map = {f.farmer_id: f for f in data.farmers}

    for fid, (selected, obj, cost) in results.items():
        farmer = farmer_map[fid]

        practice_ids = [str(data.practices[j].practice_id) for j in sorted(selected)]
        practice_names = [data.practices[j].name for j in sorted(selected)]

        rows.append({
            "farmer_id": fid,
            "farm_size_ha": farmer.farm_size_ha,
            #"budget_total_INR": round(farmer.total_budget, 2),
            "budget_total_INR": round(farmer.budget_per_ha * farmer.farm_size_ha, 2),
            "n_practices": len(selected),
            "practice_ids": "; ".join(practice_ids),
            "practices": "; ".join(practice_names),
            "objective_total_INR_per_season": round(obj, 2),
            "gross_outlay_total_INR": round(cost, 2),
        })

    df = pd.DataFrame(rows).sort_values("farmer_id").reset_index(drop=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    out_path = "results/individual_selection.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved to {out_path}")
    print("\nFirst few rows:")
    print(df.head())

    print("\nSummary:")
    print(df[[
        "farm_size_ha",
        "budget_total_INR",
        "n_practices",
        "objective_total_INR_per_season",
        "gross_outlay_total_INR"
    ]].describe())


if __name__ == "__main__":
    main()