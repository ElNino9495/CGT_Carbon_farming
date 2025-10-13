import numpy as np
import pandas as pd
import argparse
import os

def generate_carbon_market_data(n_farmers=250, seed=42):
  

    np.random.seed(seed)
    farmer_data_list = []
    
    fpo_ids = ['FPO_A', 'FPO_B', 'FPO_C']
    
    operation_practices = {

        'Laser Land Levelling': {'min_cost': 600, 'max_cost': 900},
        'Zero/Minimal Tillage': {'min_cost': 200, 'max_cost': 400},
    }
    practice_names = list(operation_practices.keys())

    farm_sizes = np.round(np.random.gamma(2, 1.5, n_farmers), 2)
    
    for i, size_acres in enumerate(farm_sizes):
        farmer_id = f"F{i+1:05d}"
        
      
        fpo_id = np.random.choice(fpo_ids, p=[0.4, 0.4, 0.2])
        operation_type = np.random.choice(practice_names, p=[0.5, 0.5])
        
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
            'Operation_Type': operation_type, # New column
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
    parser.add_argument("--filename", type=str, default="farmer_data_v2.csv" )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = generate_carbon_market_data(n_farmers=args.n_farmers, seed=args.seed)
    output_path = os.path.join(args.output_dir, args.filename)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated data for {len(df)} farmers.")
    print(f"Saved dataset to: {output_path}")
    print("\nData Summary:")
    print(f"  Mean Farm Size: {df['Farm_Size_Acres'].mean():.2f} acres")
    print("\nOperational Cost Summary by Practice Type:")
    print(df.groupby('Operation_Type')['Operational_Cost_INR'].describe().round(2))

