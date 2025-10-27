import numpy as np
import config
BASE_PRICE = 1400

def linear_coalition_value(farmers_df, params):
    """
     v(S) = (1 - δ) * [ P * Σ C_i - Σ c_i_op ]
    """
    if farmers_df.empty:
        return 0

    commission = params.get('commission', 0.2)
    total_carbon = farmers_df['Actual_Carbon_Sequestered_tCO2e'].sum()
    total_op_costs = farmers_df['Operational_Cost_INR'].sum()

    gross_project_value = BASE_PRICE * total_carbon - total_op_costs
    net_value_for_farmers = (1 - commission) * gross_project_value

    return max(0, net_value_for_farmers)


def characteristic_function_v(farmers_df, params):
    """
    
    v(S) = (1 - δ) * [ P * Σ Cᵢ - C_fixed - β|S|^θ - Σ c_opᵢ ]
    
    """
    if farmers_df.empty:
        return 0.0

    P         = config.P
    delta     = config.COMMISSION
    C_fixed   = config.FIXED_COST
    
    beta      = params.get('beta', 500)       
    theta     = params.get('theta', 0.5)      

    n = len(farmers_df)
    if n == 0:
        return 0.0

    total_carbon = farmers_df['Actual_Carbon_Sequestered_tCO2e'].sum()
    total_op_costs = farmers_df['Operational_Cost_INR'].sum()

    gross_revenue = P * total_carbon
    coalition_level_cost = beta * (n ** theta)

    value_before_commission = (
        gross_revenue - C_fixed - coalition_level_cost - total_op_costs
    )

    net_value = (1 - delta) * value_before_commission

    return max(0.0, net_value)
