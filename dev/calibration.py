# calibration.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Import our new modular files
import config
from characteristic_functions import characteristic_function_v
from game_theory_utils import monte_carlo_shapley


class CoalitionGameCalibrator:
    """
    Calibrates beta and theta parameters for the coalition characteristic function
    to ensure realistic payoffs and game convexity.
    """
    
    def __init__(self, df, sampling_strategy, output_dir):
        """
        Parameters are now pulled from config.py
        """
        self.df = df
        self.sampling_strategy = sampling_strategy
        self.output_dir = output_dir
        
        # Pull from config
        self.pi_min = config.PI_MIN
        self.pi_max = config.PI_MAX
        self.lambda_weight = config.LAMBDA_WEIGHT
        self.max_coalition_size = min(config.MAX_COALITION_SIZE, len(df))
        self.seed = config.SEED
        
        np.random.seed(self.seed)
        
        # Pre-sample coalitions based on strategy
        self.coalitions = self._create_coalitions()
        print(f"Created {len(self.coalitions)} nested coalitions for strategy '{sampling_strategy}'")

    def _create_coalitions(self):
        """
        Create *nested* coalitions of varying sizes using the specified sampling strategy
        to allow for correct convexity checking.
        """
        coalitions = {}
        
        if self.sampling_strategy == 'random':
            # --- FIX 1 (LOGIC) ---
            # 1. Create ONE single random permutation of all farmers
            shuffled_df = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            # 2. Create nested coalitions by taking the first k
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
        """Compute v(S) for a given coalition"""
        # Params from config are read inside characteristic_function_v
        params = {
            'beta': beta,
            'theta': theta
        }
        return characteristic_function_v(coalition_df, params)
    
    def _compute_shapley_values(self, coalition_df, beta, theta):
        """Compute Shapley values for a coalition"""
        params = {
            'beta': beta,
            'theta': theta
        }
        return monte_carlo_shapley(coalition_df, params, characteristic_function_v)
    
    def payoff_loss(self, beta, theta):
        """
        L_payoff(β,θ) = Σ [max(0, π̄(k) - π_max)² + max(0, π_min - π̄(k))²]
        
        --- FIX 2 (PERFORMANCE) ---
        Uses v(S)/k as a fast proxy for average payoff during optimization.
        """
        total_loss = 0.0
        
        for k in range(2, self.max_coalition_size + 1):
            coalition = self.coalitions[k]
            
            # NEW (Fast Proxy):
            v_S = self._compute_coalition_value(coalition, beta, theta)
            avg_payoff = v_S / k
            
            # Penalty for being outside target range
            upper_violation = max(0, avg_payoff - self.pi_max)
            lower_violation = max(0, self.pi_min - avg_payoff)
            
            total_loss += upper_violation**2 + lower_violation**2
            
        return total_loss
    
    def convexity_loss(self, beta, theta):
        total_loss = 0.0
        marginal_contributions = []
    
        # Start from k=2 (smallest coalition we have)
        for k in range(2, self.max_coalition_size + 1):
            coalition = self.coalitions[k]
            v_S = self._compute_coalition_value(coalition, beta, theta)
            
            if k == 2:
                # For the first coalition (k=2), marginal contribution is just v(S) - v(∅)
                # where v(∅) = 0
                marginal_contribution = v_S
            else:
                # For k > 2, marginal contribution is v(S_k) - v(S_{k-1})
                prev_coalition = self.coalitions[k-1]
                v_S_minus_1 = self._compute_coalition_value(prev_coalition, beta, theta)
                marginal_contribution = v_S - v_S_minus_1
                
            marginal_contributions.append(marginal_contribution)
        
        # Check for violations of non-decreasing marginal contributions
        # We want Δv(k) ≤ Δv(k+1) for all k
        for i in range(len(marginal_contributions) - 1):
            violation = max(0, marginal_contributions[i] - marginal_contributions[i+1])
            total_loss += violation**2
            
        return total_loss

    
    def joint_loss(self, x):
        """
        L(β,θ) = L_payoff(β,θ) + λ * L_convexity(β,θ)
        """
        beta, theta = x
        
        if beta <= 0 or theta <= 0 or theta >= 1:
            return 1e10  # Large penalty
        
        L_pay = self.payoff_loss(beta, theta)
        L_conv = self.convexity_loss(beta, theta)
        
        return L_pay + self.lambda_weight * L_conv
    
    def optimize_grid_search(self):
        """Perform grid search over beta and theta ranges from config"""
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
        """Use scipy optimization methods from config"""
        
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
        """
        Analyze a solution (beta, theta) in detail.
        This is where we use the SLOW, ACCURATE Shapley values.
        """
        print(f"\nAnalyzing final solution (β={beta:.1f}, θ={theta:.3f}) using Monte Carlo Shapley...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = []
        for k in tqdm(range(2, self.max_coalition_size + 1), desc="Analyzing solution"):
            coalition = self.coalitions[k]
            
            # Use accurate Shapley values for final analysis
            shapley_values = self._compute_shapley_values(coalition, beta, theta)
            v_S = self._compute_coalition_value(coalition, beta, theta)
            
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
        
        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Calibration Analysis for Strategy: {self.sampling_strategy}', fontsize=16)

        # 1. Average payoff vs coalition size
        ax = axes[0, 0]
        ax.plot(results_df['coalition_size'], results_df['avg_payoff'], 'b-', linewidth=2)
        ax.axhline(self.pi_min, color='r', linestyle='--', label=f'Min target ({self.pi_min:,.0f})')
        ax.axhline(self.pi_max, color='r', linestyle='--', label=f'Max target ({self.pi_max:,.0f})')
        ax.fill_between(results_df['coalition_size'], self.pi_min, self.pi_max, alpha=0.2, color='green')
        ax.set_title(f'Average Payoff vs Size (β={beta:.1f}, θ={theta:.3f})')
        ax.legend()
        
        # 2. Coalition value vs size
        ax = axes[0, 1]
        ax.plot(results_df['coalition_size'], results_df['coalition_value'], 'g-', linewidth=2)
        ax.set_title('Coalition Value v(S) vs Size')

        # 3. Marginal contributions
        ax = axes[1, 0]
        marginal = results_df['coalition_value'].diff()
        ax.plot(results_df['coalition_size'][1:], marginal[1:], 'o-', linewidth=2)
        ax.set_title('Marginal Contributions (Test for Convexity)')
        
        # 4. Payoff range (min/max)
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
        
        # Save results
        results_df.to_csv(f'{self.output_dir}/calibration_results_{self.sampling_strategy}.csv', index=False)
        
        print(f"\nAnalysis saved to {self.output_dir}")
        print(f"  Coalitions in target range: {results_df['in_target_range'].sum()}/{len(results_df)}")
        print(f"  Average payoff range: {results_df['avg_payoff'].min():.0f} - {results_df['avg_payoff'].max():.0f} Rs")
        
        return results_df