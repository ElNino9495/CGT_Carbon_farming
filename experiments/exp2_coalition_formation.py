"""
Experiment 2: Coalition formation analysis.
  - Compute characteristic function for all sub-coalitions
  - Shapley values
  - Core check
  - Diversity-efficiency tradeoff
"""
import sys
sys.path.insert(0, ".")

from src.config import ModelConfig
from src.data_loader import load_data
from src.practice_selection import solve_all_farmers
from src.coalition import CoalitionGame
from src.solution_concepts import shapley_value, shapley_value_sampling, is_in_core, nucleolus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    config = ModelConfig(CCP=1500.0, CROP_PRICE=22000.0, solver="gurobi")
    data = load_data("data/synthetic")

    # Step 1: Solve individual selections
    print("Solving individual practice selections...")
    ind_results = solve_all_farmers(data, config)
    portfolios = {fid: sel for fid, (sel, _, _) in ind_results.items()}

    # Step 2: Form coalition game
    game = CoalitionGame(data, config, portfolios)

    # Select a subset for exact analysis (Shapley is O(2^n))
    # For n <= 15, exact is feasible; for larger n, use sampling
    farmer_subset = [f.farmer_id for f in data.farmers[:10]]  # first 10
    grand_coalition = frozenset(farmer_subset)

    print(f"\n=== Coalition analysis for {len(farmer_subset)} farmers ===")

    # Singleton values
    print("\nSingleton values:")
    for fid in farmer_subset:
        sv = game.singleton_value(fid)
        print(f"  {fid}: v_tilde = {sv:.2f} INR")

    # Grand coalition value
    v_grand = game.characteristic_value(grand_coalition)
    print(f"\nGrand coalition value v(N) = {v_grand:.2f} INR")
    print(f"Practice diversity |U_S| = {game.practice_diversity(grand_coalition)}")

    surplus = game.coalition_surplus(grand_coalition)
    print(f"Coalition surplus = {surplus:.2f} INR")

    # Shapley values
    n = len(farmer_subset)
    if n <= 18:
        print(f"\nComputing exact Shapley values (n={n})...")
        phi = shapley_value(game, grand_coalition)
    else:
        print(f"\nComputing approximate Shapley values (n={n}, 10000 samples)...")
        phi = shapley_value_sampling(game, grand_coalition, n_samples=10000)

    print("\nShapley allocation:")
    for fid in farmer_subset:
        farmer = next(f for f in data.farmers if f.farmer_id == fid)
        print(f"  {fid} (size={farmer.farm_size_ha:.2f}ha): "
              f"phi={phi[fid]:.2f}, singleton={game.singleton_value(fid):.2f}, "
              f"IR={'✓' if phi[fid] >= game.singleton_value(fid) else '✗'}")

    print(f"\nSum of Shapley = {sum(phi.values()):.2f}, v(N) = {v_grand:.2f}")

    # Core check
    if n <= 15:
        in_core = is_in_core(game, phi, grand_coalition)
        print(f"Shapley in core: {in_core}")

    # Diversity-efficiency tradeoff
    print("\n=== Diversity-Efficiency Tradeoff ===")
    for k in range(2, min(n + 1, 8)):
        from itertools import combinations
        best_surplus = -float("inf")
        best_coalition = None
        for combo in combinations(farmer_subset, k):
            S = frozenset(combo)
            s = game.coalition_surplus(S)
            if s > best_surplus:
                best_surplus = s
                best_coalition = S
        if best_coalition:
            div = game.practice_diversity(best_coalition)
            print(f"  k={k}: best surplus={best_surplus:.2f}, diversity={div}")


if __name__ == "__main__":
    main()