import numpy as np
import pandas as pd
import argparse
import os

def generate_carbon_market_data(n_farmers=250, seed=42):
    np.random.seed(seed)
    farmer_data_list = []
    
    fpo_ids = ['FPO_A', 'FPO_B', 'FPO_C']
    
    operation_practices = {
        'Lazer Land Leveling': {'cost': 400, 'adoption': 1.4},
        'Optimized Fertilizer Use': {'cost': 300, 'adoption': 20.0},
        'Soil Amendments': {'cost': 500, 'adoption': 2.5},
        'Improved Water Management (AWD)': {'cost': 600, 'adoption': 14.3},
        'Diversified Cropping (Cover/Rotation)': {'cost': 450, 'adoption': 21.4},
        'Reduced Tillage + Optimized Fertilizer Use': {'cost': 650, 'adoption': 10.7},
        'Reduced Tillage + Soil Amendments + Water Management': {'cost': 1350, 'adoption': 6.07},
        'Fertilizer Use + Soil Amendments + Cropping Diversification': {'cost': 1150, 'adoption': 14.63},
        'All Five Practices Combined': {'cost': 1800, 'adoption': 11.92},
        'Water Management + Cropping Diversification': {'cost': 950, 'adoption': 17.85}
    }

   
    for practice in operation_practices.values():
        practice['min_cost'] = practice['cost'] * 0.9
        practice['max_cost'] = practice['cost'] * 1.1

    practice_names = list(operation_practices.keys())
    
   
    adoption_rates = np.array([p['adoption'] for p in operation_practices.values()])
    probabilities = adoption_rates / adoption_rates.sum()

    mu_hectares = -0.5
    sigma_hectares = 1.0
    
    farm_sizes_hectares = np.random.lognormal(mean=mu_hectares, sigma=sigma_hectares, size=n_farmers)
    
    HECTARES_TO_ACRES = 2.47105
    farm_sizes_acres = farm_sizes_hectares * HECTARES_TO_ACRES
    
    farm_sizes_acres = np.round(farm_sizes_acres, 2)
    
    for i, size_acres in enumerate(farm_sizes_acres):
        farmer_id = f"F{i+1:05d}"
        
        fpo_id = np.random.choice(fpo_ids, p=[0.4, 0.4, 0.2])
        
        operation_type = np.random.choice(practice_names, p=probabilities)
        
        base_carbon_per_acre = 6
        carbon_variation = np.random.normal(1.0, 0.3)
        carbon_sequestered = max(0, np.round(base_carbon_per_acre * size_acres * carbon_variation, 2))
        
        practice_info = operation_practices[operation_type]
        cost_per_acre = np.random.uniform(practice_info['min_cost'], practice_info['max_cost'])
        operational_cost = np.round(cost_per_acre * size_acres, -2)
        
        farmer_data = {
            'Farmer_ID': farmer_id,
            'Farm_Size_Acres': size_acres,
            'FPO_ID': fpo_id,
            'Operation_Type': operation_type,
            'Operational_Cost_INR': operational_cost,
            'Actual_Carbon_Sequestered_tCO2e': carbon_sequestered,
            'Carbon_per_Acre': np.round(carbon_sequestered / size_acres, 2) if size_acres > 0 else 0
        }
        farmer_data_list.append(farmer_data)
        
    return pd.DataFrame(farmer_data_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic farmer data for VCM simulations.")
    parser.add_argument("-n", "--n_farmers", type=int, default=250)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output_dir", type=str, default="./data/synthetic/")
    parser.add_argument("--filename", type=str, default="farmer_data_log_normal.csv" )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = generate_carbon_market_data(n_farmers=args.n_farmers, seed=args.seed)
    output_path = os.path.join(args.output_dir, args.filename)
    df.to_csv(output_path, index=False)
    
 