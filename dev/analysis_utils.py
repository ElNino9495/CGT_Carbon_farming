import pandas as pd
import numpy as np
from game_theory_utils import monte_carlo_shapley, approximate_core_stability, is_game_convex
from characteristic_functions import BASE_PRICE

def calculate_cba_metrics(coalition_df, params):
    if coalition_df.empty:
        return {'aggregator_profit': 0, 'roi': 0, 'total_project_profit': 0}
        
    commission = params.get('commission', 0.2)
    total_carbon = coalition_df['Actual_Carbon_Sequestered_tCO2e'].sum()
    total_op_costs = coalition_df['Operational_Cost_INR'].sum()
    
    gross_revenue = BASE_PRICE * total_carbon
    project_profit = gross_revenue - total_op_costs
    
    aggregator_profit = commission * max(0, project_profit)
    roi = (project_profit / total_op_costs) if total_op_costs > 0 else float('inf')
    
    return {'aggregator_profit': aggregator_profit, 'roi': roi, 'total_project_profit': project_profit}

def compute_game_metrics(coalition_df, params, value_function, return_shapley_dict=False):
    if coalition_df.empty:
        base_result = {'avg_shapley': 0, 'core_stable': True, 'blocking_probability': 0.0, 'is_convex': True, 'convexity_violation_rate': 0.0}
        return (base_result, {}) if return_shapley_dict else base_result

    shapley_values = monte_carlo_shapley(coalition_df, params, value_function)
    
    is_stable, blocking_prob = approximate_core_stability(coalition_df, shapley_values, params, value_function)
    is_convex, convex_vio_rate = is_game_convex(coalition_df, params, value_function)

    # analyze_heterogeneity has been removed as it complicates the current structure.
    # It can be added back later if needed.
    result = {
        'avg_shapley': np.mean(list(shapley_values.values())) if shapley_values else 0,
        'core_stable': is_stable,
        'blocking_probability': blocking_prob,
        'is_convex': is_convex,
        'convexity_violation_rate': convex_vio_rate,
    }
    return (result, shapley_values) if return_shapley_dict else result