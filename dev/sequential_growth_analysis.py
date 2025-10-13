import pandas as pd
import argparse
import os
from tqdm import tqdm
import numpy as np
import time

from analysis_utils import calculate_cba_metrics, compute_game_metrics
from visualization import visualize_growth_trends
from characteristic_functions import linear_coalition_value, exponential_synergy_value, characteristic_function_v

VALUE_FUNCTIONS = {
    'linear': linear_coalition_value,
    'synergy': exponential_synergy_value,
    'new': characteristic_function_v
}

def run_sequential_growth(fpo_df, max_size, params, value_function, seed):
    if len(fpo_df) < max_size:
        return []

    farmers_for_run = fpo_df.sample(n=max_size, random_state=seed)
    results = []

    for k in range(2, max_size + 1):
        current_coalition_df = farmers_for_run.head(k)

        game_metrics, shapley_values = compute_game_metrics(
            current_coalition_df, params, value_function, return_shapley_dict=True
        )
        cba_metrics = calculate_cba_metrics(current_coalition_df, params)

        for farmer_id, shap_val in shapley_values.items():
            res = {
                'num_farmers': k,
                'farmer_id': farmer_id,
                'farmer_shapley': shap_val,
                'actual_area': current_coalition_df['Farm_Size_Acres'].sum(),
                **game_metrics,
                **cba_metrics
            }
            results.append(res)
    return results

def main():
    parser = argparse.ArgumentParser(description="Sequential growth analysis of farmer coalitions.")
    parser.add_argument("--model", type=str, default="linear", choices=['linear', 'synergy', 'new'], help="Select the characteristic function model to use.")
    parser.add_argument("--commission", type=float, default=0.2, help="Aggregator commission rate.")
    parser.add_argument("--beta", type=float, default=0.05, help="Synergy coefficient (strength).")
    parser.add_argument("--theta", type=float, default=1.2, help="Synergy exponent (growth rate).")
    parser.add_argument("--training_cost", type=float, default=50000, help="Fixed training cost for the coalition.")
    parser.add_argument("--fpo_id", type=str, default="ALL", help="Specific FPO_ID to analyze.")
    parser.add_argument("--runs", type=int, default=50, help="Number of simulation runs.")
    parser.add_argument("--max_size", type=int, default=30, help="Maximum coalition size.")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed.")
    parser.add_argument("--datafile", type=str, default="./data/synthetic/farmer_data_v3.csv")
    args = parser.parse_args()
    
    params = {
        'commission': args.commission,
        'beta': args.beta,
        'theta': args.theta,
        'training_cost': args.training_cost
    }
    
    value_function = VALUE_FUNCTIONS[args.model]
    print(f"--- Running simulation with '{args.model}' model ---")

    df = pd.read_csv(args.datafile)
    all_results = []

    if args.fpo_id.upper() == 'ALL':
        fpo_to_analyze = df['FPO_ID'].unique()
    else:
        fpo_to_analyze = [args.fpo_id]

    print(f"Analyzing FPOs: {', '.join(fpo_to_analyze)}")

    for fpo_id in tqdm(fpo_to_analyze, desc="Processing FPOs"):
        fpo_specific_df = df[df['FPO_ID'] == fpo_id].copy()
        if len(fpo_specific_df) < args.max_size:
            continue
        for run_num in tqdm(range(args.runs), desc=f"Runs for {fpo_id}", leave=False):
            run_seed = args.seed + run_num
            run_results = run_sequential_growth(fpo_specific_df, args.max_size, params, value_function, seed=run_seed)
            for res in run_results:
                res['run_id'] = run_num
                res['fpo_id'] = fpo_id
            all_results.extend(run_results)

    if not all_results:
        print("No results generated.")
        return

    results_df = pd.DataFrame(all_results)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'./data/results/{args.model}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults data saved to {results_path}")

    log_path = os.path.join(output_dir, 'run_parameters.log')
    with open(log_path, 'w') as f:
        f.write(f"Simulation Run Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("-" * 30 + "\n")
        
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    visualize_growth_trends(results_df, output_dir)

if __name__ == "__main__":
    main()