import numpy as np

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


def exponential_synergy_value(farmers_df, params):
    """
    v(S) = (1 - δ) * [ P Σ C_i (1 + β |S|^θ)  -  Σ c_i^op  -  T_train / |S| ]
    where T_train is the **fixed** total training cost that is **equally split**
    among the coalition members.
    """
    if farmers_df.empty:
        return 0.0

    # unpack parameters
    P       = params.get('P', 1400)           # market price (INR / tCO₂e)
    delta   = params.get('commission', 0.30)  # aggregator + FPO cut
    beta    = params.get('beta', 0.05)        # network-effect strength
    theta   = params.get('theta', 1.2)        # network-effect exponent
    T_train = params.get('training_cost', 50000)  # fixed training bill

    n            = len(farmers_df)
    total_carbon = farmers_df['Actual_Carbon_Sequestered_tCO2e'].sum()
    total_op     = farmers_df['Operational_Cost_INR'].sum()

    # 1. gross revenue with network synergy
    gross_revenue = P * total_carbon * (1 + beta * (n ** theta))

    # 2. subtract operating costs and shared training cost
    net_before_commission = gross_revenue - total_op - (T_train / n)

    # 3. take out aggregator/FPO commission
    v_S = (1 - delta) * max(0.0, net_before_commission)

    return v_S

def characteristic_function_v(farmers_df, params):
    """
    Calculates the value of a coalition based on the formula:
    v(S) = (1 - δ) * [ P * Σ Cᵢ - C_fixed - β|S|^θ - Σ c_opᵢ ]
    """
    if farmers_df.empty:
        return 0.0

    # Unpack parameters from the dictionary, with default values
    P         = params.get('P', 1400)          # Market price (Rs/tCO2e)
    delta     = params.get('commission', 0.3)  # Aggregator commission rate (δ)
    C_fixed   = params.get('fixed_cost', 30000)# Fixed training/setup cost
    beta      = params.get('beta', 500)        # Coalition-level cost coefficient (β)
    theta     = params.get('theta', 0.5)       # Coalition-level cost exponent (θ)

    # Get the size of the coalition |S|
    n = len(farmers_df)

    # Calculate Σ Cᵢ and Σ c_opᵢ from the dataframe
    total_carbon = farmers_df['Actual_Carbon_Sequestered_tCO2e'].sum()
    total_op_costs = farmers_df['Operational_Cost_INR'].sum()

    # Calculate gross revenue: P * Σ Cᵢ
    gross_revenue = P * total_carbon

    # Calculate coalition-level costs: β|S|^θ
    coalition_level_cost = beta * (n ** theta)

    # Calculate the total value before commission
    value_before_commission = (
        gross_revenue - C_fixed - coalition_level_cost - total_op_costs
    )

    # Apply the aggregator commission: (1 - δ) * [...]
    net_value = (1 - delta) * value_before_commission

    # The value of a coalition cannot be negative
    return max(0.0, net_value)



