import pandas as pd
import os
import config
from calibration import CoalitionGameCalibrator 
import time
from datetime import datetime

def main():
    
    data_path = os.path.join(config.DATA_DIR, config.DATA_FILENAME)
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} farmer records from {data_path}")
    
    strategies = config.STRATEGIES
    all_results = {}
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_results_dir = os.path.join(config.RESULTS_DIR, timestamp)
    os.makedirs(timestamped_results_dir, exist_ok=True)
    print(f"\nResults will be saved to: {timestamped_results_dir}")
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"CALIBRATING WITH {strategy.upper()} SAMPLING STRATEGY")
        print(f"{'='*60}")
        
        strategy_output_dir = os.path.join(timestamped_results_dir, strategy)
        os.makedirs(strategy_output_dir, exist_ok=True)
        
        calibrator = CoalitionGameCalibrator(
            df=df,
            sampling_strategy=strategy,
            output_dir=strategy_output_dir
        )
        
        _, best_params_grid = calibrator.optimize_grid_search()
        
        scipy_result = calibrator.optimize_scipy(initial_guess=best_params_grid)
        
        beta_final, theta_final = scipy_result.x
        analysis_df = calibrator.analyze_solution(beta_final, theta_final)
        
        all_results[strategy] = {
            'beta': beta_final,
            'theta': theta_final,
            'loss': scipy_result.fun,
            'analysis': analysis_df
        }
    
    print(f"\n{'='*60}")
    print("FINAL COMPARISON OF SAMPLING STRATEGIES")
    print(f"{'='*60}")
    
    comparison_data = []
    for strategy, results in all_results.items():
        avg_payoff_min = results['analysis']['avg_payoff'].min()
        avg_payoff_max = results['analysis']['avg_payoff'].max()
        
   

        comparison_data.append({
            'Strategy': strategy,
            'Beta': results['beta'],
            'Theta': results['theta'],
            'Total Loss': results['loss'],
            'Avg Payoff Range': f"{avg_payoff_min:.0f} - {avg_payoff_max:.0f}"
            
        })
        
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(
        os.path.join(timestamped_results_dir, 'strategy_comparison.csv'), 
        index=False
    )
    print(f"\nSaved final comparison to {timestamped_results_dir}")


if __name__ == "__main__":
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    main()