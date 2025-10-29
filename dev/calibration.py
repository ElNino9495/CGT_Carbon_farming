import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
    
import config
from characteristic_functions import characteristic_function_v
from game_theory_utils import monte_carlo_shapley


class CoalitionGameCalibrator:
   
    
    def __init__(self, df, sampling_strategy, output_dir):
      
        self.df = df
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir
        
        self.pi_min = config.PI_MIN
        self.pi_max = config.PI_MAX
        self.lambda_weight = config.LAMBDA_WEIGHT
        self.max_coalition_size = min(config.MAX_COALITION_SIZE, len(df))
        self.seed = config.SEED
        
        np.random.seed(self.seed)
        
        self.coalitions = self._create_coalitions()
        print(f"Created {len(self.coalitions)} nested coalitions for strategy '{sampling_strategy}'")

    def _create_coalitions(self):
     
        coalitions = {}
        
        if self.sampling_strategy == 'random':
           
            shuffled_df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            for k in range(2, self.max_coalition_size + 1):
                coalitions[k] = shuffled_df.head(k)
                
        elif self.sampling_strategy == 'ascending':
            sorted_df = self.df.sort_values('Farm_Size_Acres', ascending=True).reset_index(drop=True)
            for k in range(2, self.max_coalition_size + 1):
                coalitions[k] = sorted_df.head(k)
                
        elif self.sampling_strategy == 'descending':
            sorted_df = self.df.sort_values('Farm_Size_Acres', ascending=False).reset_index(drop=True)
            for k in range(2, self.max_coalition_size + 1):
                coalitions[k] = sorted_df.head(k)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
        return coalitions
    
    def _compute_coalition_value(self, coalition_df, beta, theta):
        params = {
            'beta': beta,
            'theta': theta
        }
        return characteristic_function_v(coalition_df, params)
    

    
    def _compute_shapley_values(self, coalition_df, beta, theta):
        params = {
            'beta': beta,
            'theta': theta
        }
        return monte_carlo_shapley(coalition_df, params, characteristic_function_v)
    
    def payoff_loss(self, beta, theta):
      
        total_loss = 0.0
        
        for k in range(2, self.max_coalition_size + 1):
            coalition = self.coalitions[k]
            
            v_S = self._compute_coalition_value(coalition, beta, theta)
            avg_payoff = v_S / k
            
            upper_violation = max(0, avg_payoff - self.pi_max)
            lower_violation = max(0, self.pi_min - avg_payoff)
            
            total_loss += upper_violation**2 + lower_violation**2
            
        return total_loss
    
    def convexity_loss(self, beta, theta):
        total_loss = 0.0
        marginal_contributions = []
    
        for k in range(2, self.max_coalition_size + 1):
            coalition = self.coalitions[k]
            v_S = self._compute_coalition_value(coalition, beta, theta)
            
            if k == 2:
               
                marginal_contribution = v_S
            else:
                prev_coalition = self.coalitions[k-1]
                v_S_minus_1 = self._compute_coalition_value(prev_coalition, beta, theta)
                marginal_contribution = v_S - v_S_minus_1
                
            marginal_contributions.append(marginal_contribution)
        
      
        for i in range(len(marginal_contributions) - 1):
            violation = max(0, marginal_contributions[i] - marginal_contributions[i+1])
            total_loss += violation**2
            
        return total_loss

    
    def joint_loss(self, x):
        
        beta, theta = x
        
        if beta <= 0 or theta <= 0 or theta >= 1:
            return 1e10 
        
        L_pay = self.payoff_loss(beta, theta)
        L_conv = self.convexity_loss(beta, theta)
        
        return L_pay + self.lambda_weight * L_conv
    
    def optimize_grid_search(self):
        beta_range = config.GRID_BETA_RANGE
        theta_range = config.GRID_THETA_RANGE
        
        results = []
        best_loss = float('inf')
        best_params = None
        
        print(f"\nGrid search with {len(beta_range)} x {len(theta_range)} = {len(beta_range)*len(theta_range)} combinations")
        
        for beta in tqdm(beta_range, desc="Beta values"):
            for theta in theta_range:
                loss = self.joint_loss([beta, theta])
                L_pay = self.payoff_loss(beta, theta)
                L_conv = self.convexity_loss(beta, theta)
                
                results.append({
                    'beta': beta, 'theta': theta, 'total_loss': loss,
                    'payoff_loss': L_pay, 'convexity_loss': L_conv
                })
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (beta, theta)
        
        results_df = pd.DataFrame(results)
        
        print(f"\nBest parameters from grid search:")
        print(f"  Beta: {best_params[0]:.2f}, Theta: {best_params[1]:.4f}, Loss: {best_loss:.2f}")
        
        return results_df, best_params
    
    def optimize_scipy(self, initial_guess):
        
        print(f"\nOptimizing using {config.SCIPY_METHOD}...")
        result = minimize(
            self.joint_loss,
            x0=initial_guess,
            method=config.SCIPY_METHOD,
            bounds=config.SCIPY_BOUNDS
        )
        
        beta_opt, theta_opt = result.x
        print(f"\nOptimization result:")
        print(f"  Beta: {beta_opt:.2f}, Theta: {theta_opt:.4f}, Loss: {result.fun:.2f}, Success: {result.success}")
        
        return result
    


    

    def analyze_solution(self, beta, theta):
      
        print(f"\nAnalyzing final solution (β={beta:.1f}, θ={theta:.3f}) using Monte Carlo Shapley...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = []
        all_shapley_data = []
        for k in tqdm(range(2, self.max_coalition_size + 1), desc="Analyzing solution"):
            coalition = self.coalitions[k]
            
            # Use accurate Shapley values for final analysis
            shapley_values = self._compute_shapley_values(coalition, beta, theta)
            v_S = self._compute_coalition_value(coalition, beta, theta)

            for farmer_id, earning in shapley_values.items():
                all_shapley_data.append({
                    'coalition_size': k,
                    'Farmer_ID': farmer_id,
                    'shapley_earning': earning
                })
            
            avg_payoff = np.mean(list(shapley_values.values()))
            min_payoff = min(shapley_values.values())
            max_payoff = max(shapley_values.values())
            
            results.append({
                'coalition_size': k, 'coalition_value': v_S,
                'avg_payoff': avg_payoff, 'min_payoff': min_payoff,
                'max_payoff': max_payoff,
                'in_target_range': self.pi_min <= avg_payoff <= self.pi_max
            })
        
        results_df = pd.DataFrame(results)

        shapley_df = pd.DataFrame(all_shapley_data)
        shapley_csv_path = f'{self.output_dir}/individual_shapley_earnings_{self.sampling_strategy}.csv'
        shapley_df.to_csv(shapley_csv_path, index=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Calibration Analysis for Strategy: {self.sampling_strategy}', fontsize=16)

        ax = axes[0, 0]
        ax.plot(results_df['coalition_size'], results_df['avg_payoff'], 'b-', linewidth=2)
        ax.axhline(self.pi_min, color='r', linestyle='--', label=f'Min target ({self.pi_min:,.0f})')
        ax.axhline(self.pi_max, color='r', linestyle='--', label=f'Max target ({self.pi_max:,.0f})')
        ax.fill_between(results_df['coalition_size'], self.pi_min, self.pi_max, alpha=0.2, color='green')
        ax.set_title(f'Average Payoff vs Size (β={beta:.1f}, θ={theta:.3f})')
        ax.legend()
        
        ax = axes[0, 1]
        ax.plot(results_df['coalition_size'], results_df['coalition_value'], 'g-', linewidth=2)
        ax.set_title('Coalition Value v(S) vs Size')

        ax = axes[1, 0]
        marginal = results_df['coalition_value'].diff()
        ax.plot(results_df['coalition_size'][1:], marginal[1:], 'o-', linewidth=2)
        ax.set_title('Marginal Contributions (Test for Convexity)')
        
        ax = axes[1, 1]
        ax.plot(results_df['coalition_size'], results_df['min_payoff'], 'r-', label='Min Shapley', linewidth=2, alpha=0.7)
        ax.plot(results_df['coalition_size'], results_df['avg_payoff'], 'b-', label='Avg Shapley', linewidth=2)
        ax.plot(results_df['coalition_size'], results_df['max_payoff'], 'g-', label='Max Shapley', linewidth=2, alpha=0.7)
        ax.set_title('Shapley Payoff Distribution')
        ax.legend()

        for ax_row in axes:
            for ax in ax_row:
                ax.grid(alpha=0.3)
                ax.set_xlabel('Coalition Size')
                ax.set_ylabel('Value (Rs)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{self.output_dir}/calibration_analysis_{self.sampling_strategy}.png', dpi=300)
        plt.close()
        
        results_df.to_csv(f'{self.output_dir}/calibration_results_{self.sampling_strategy}.csv', index=False)
        
        print(f"\nAnalysis saved to {self.output_dir}")
        print(f"  Coalitions in target range: {results_df['in_target_range'].sum()}/{len(results_df)}")
        print(f"  Average payoff range: {results_df['avg_payoff'].min():.0f} - {results_df['avg_payoff'].max():.0f} Rs")
        
        return results_df