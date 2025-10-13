import numpy as np
import pandas as pd
import random
from itertools import combinations, permutations

def exact_shapley(farmers_df, params, value_function):
    """
    Computes exact Shapley values for a given value_function.
    """
    n = len(farmers_df)
    if n == 0:
        return {}

    shapley_values = {f: 0 for f in farmers_df['Farmer_ID']}
    all_ids = list(farmers_df['Farmer_ID'])
    farmers_by_id = farmers_df.set_index('Farmer_ID')

    for perm in permutations(all_ids):
        prev_value = 0
        for i in range(1, n + 1):
            player_id = perm[i - 1]
            current_players_df = farmers_by_id.loc[list(perm[:i])]
            value_after = value_function(current_players_df, params)
            marginal_contribution = value_after - prev_value
            shapley_values[player_id] += marginal_contribution
            prev_value = value_after

    total_perms = len(list(permutations(all_ids)))
    for fid in shapley_values:
        shapley_values[fid] /= total_perms
    return shapley_values


def monte_carlo_shapley(farmers_df, params, value_function, samples=2000):
    """
    Estimates Shapley values for a given value_function.
    """
    n = len(farmers_df)
    if n == 0:
        return {}
    if n <= 7:
        return exact_shapley(farmers_df, params, value_function)

    shapley_values = {f: 0 for f in farmers_df['Farmer_ID']}
    all_ids = list(farmers_df['Farmer_ID'])
    farmers_by_id = farmers_df.set_index('Farmer_ID')

    for _ in range(samples):
        perm = random.sample(all_ids, n)
        prev_value = 0
        for i in range(1, n + 1):
            player_id = perm[i - 1]
            current_players_df = farmers_by_id.loc[perm[:i]]
            value_after = value_function(current_players_df, params)
            marginal_contribution = value_after - prev_value
            shapley_values[player_id] += marginal_contribution
            prev_value = value_after

    for fid in shapley_values:
        shapley_values[fid] /= samples

    grand_value = value_function(farmers_df, params)
    total_shapley = sum(shapley_values.values())
    if total_shapley > 0:
        factor = grand_value / total_shapley
        for k in shapley_values:
            shapley_values[k] *= factor
    return shapley_values


def approximate_core_stability(farmers_df, shapley_values, params, value_function, samples=1000, tol=1e-6):
    """
    Checks core stability for a given value_function, using exact enumeration for n<=7.
    """
    n = len(farmers_df)
    if n < 2 or not shapley_values:
        return True, 0.0

    all_ids = list(farmers_df['Farmer_ID'])
    farmers_by_id = farmers_df.set_index('Farmer_ID')
    blocking_coalitions_found = 0
    total_checks = 0

    sub_coalition_iterator = None
    if n <= 7:
        # Create an iterator for all 2^n - 2 non-trivial sub-coalitions
        all_combinations = [combinations(all_ids, size) for size in range(1, n)]
        sub_coalition_iterator = (combo for group in all_combinations for combo in group)
        total_checks = (2**n) - 2
    else:
        # Create an iterator for n random samples
        sub_coalition_iterator = (tuple(random.sample(all_ids, random.randint(1, n - 1))) for _ in range(samples))
        total_checks = samples

    for sub_coalition_ids in sub_coalition_iterator:
        sub_df = farmers_by_id.loc[list(sub_coalition_ids)]
        v_S = value_function(sub_df, params)
        shapley_sum_S = sum(shapley_values[fid] for fid in sub_coalition_ids)
        if shapley_sum_S + tol < v_S:
            blocking_coalitions_found += 1

    blocking_probability = blocking_coalitions_found / total_checks if total_checks > 0 else 0
    return blocking_probability == 0, blocking_probability


def is_game_convex(farmers_df, params, value_function, samples=1000, tol=1e-6):
    """
    Checks convexity for a given value_function, using exact enumeration for n<=7.
    """
    n = len(farmers_df)
    if n < 2:
        return True, 0.0

    all_ids = list(farmers_df['Farmer_ID'])
    farmers_by_id = farmers_df.set_index('Farmer_ID')
    violations = 0
    total_checks = 0

    if n <= 7:
        subsets_map = {}
        for r in range(n + 1):
            for combo in combinations(all_ids, r):
                # Ensure the key is always sorted
                sorted_combo = tuple(sorted(combo))
                sub_df = farmers_by_id.loc[list(sorted_combo)] if sorted_combo else pd.DataFrame()
                subsets_map[sorted_combo] = value_function(sub_df, params)
        
        subset_keys = list(subsets_map.keys())
        for i in range(len(subset_keys)):
            for j in range(i, len(subset_keys)):
                S_tuple = subset_keys[i]
                T_tuple = subset_keys[j]
                
                S_ids = set(S_tuple)
                T_ids = set(T_tuple)
                
                union_tuple = tuple(sorted(list(S_ids.union(T_ids))))
                intersect_tuple = tuple(sorted(list(S_ids.intersection(T_ids))))

                v_S = subsets_map[S_tuple]
                v_T = subsets_map[T_tuple]
                v_union = subsets_map[union_tuple]
                v_intersect = subsets_map[intersect_tuple]

                if v_S + v_T > v_union + v_intersect + tol:
                    violations += 1
                total_checks += 1
    else:
        for _ in range(samples):
            s_size = random.randint(1, n)
            t_size = random.randint(1, n)
            S_ids = set(random.sample(all_ids, s_size))
            T_ids = set(random.sample(all_ids, t_size))
            
            S_union_T_ids = S_ids.union(T_ids)
            S_intersect_T_ids = S_ids.intersection(T_ids)

            df_S = farmers_by_id.loc[list(S_ids)]
            df_T = farmers_by_id.loc[list(T_ids)]
            df_union = farmers_by_id.loc[list(S_union_T_ids)]
            df_intersect = farmers_by_id.loc[list(S_intersect_T_ids)] if S_intersect_T_ids else pd.DataFrame()

            v_S = value_function(df_S, params)
            v_T = value_function(df_T, params)
            v_union = value_function(df_union, params)
            v_intersect = value_function(df_intersect, params)

            if v_S + v_T > v_union + v_intersect + tol:
                violations += 1
            total_checks += 1

    violation_rate = violations / total_checks if total_checks > 0 else 0
    return violation_rate < 0.01, violation_rate