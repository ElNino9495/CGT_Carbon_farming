"""
Experiment 1: Solve optimal practice selection for all farmers.
"""
import sys
sys.path.insert(0, ".")

from src.config import ModelConfig
from src.data_loader import load_data
from src.practice_selection import solve_all_farmers

import pandas as pd


def main():
    config = ModelConfig(
        CCP=1500.0,
        CROP_PRICE=22000.0,
        solver="gurobi",  # change to "pulp" if no Gurobi license
        time_limit=60,
    )

    print("Loading data...")
    data = load_data("data/synthetic")
    print(f"  {len(data.farmers)} farmers, {len(data.practices)} practices")
    print(f"  {len(data.alpha)} alpha pairs, {len(data.gamma)} gamma pairs, "
          f"{len(data.delta)} delta pairs")
    print(f"  {len(data.incompatible)} incompatible pairs")

    print("\nSolving optimal practice selection for each farmer...")
    results = solve_all_farmers(data, config)

    # Save results
    rows = []
    for fid, (selected, obj, cost) in results.items():
        farmer = next(f for f in data.farmers if f.farmer_id == fid)
        practice_names = [data.practices[j].name for j in sorted(selected)]
        rows.append({
            "farmer_id": fid,
            "farm_size_ha": farmer.farm_size_ha,
            "budget": farmer.budget_per_ha,
            "n_practices": len(selected),
            "practices": "; ".join(practice_names),
            "objective_INR": round(obj, 2),
            "cost_INR": round(cost, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/individual_selection.csv", index=False)
    print(f"\nSaved to results/individual_selection.csv")
    print(df.describe())


if __name__ == "__main__":
    main()