import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
from functools import partial
from scipy.optimize import minimize

from characteristic_functions import exponential_synergy_value
from game_theory_utils import is_game_convex


def test_params(param_tuple, farmer_samples, commission, training_cost, samples):
    """Test a single beta-theta parameter combination across different coalition sizes"""
    beta, theta = param_tuple
    results = []
    for coalition_size, sample_df in farmer_samples.items():
        params = {
            'commission': commission,
            'beta': beta,
            'theta': theta,
            'training_cost': training_cost
        }
        is_convex, violation_rate = is_game_convex(
            sample_df, params, exponential_synergy_value, samples=samples
        )
        results.append({
            'beta': beta,
            'theta': theta,
            'coalition_size': coalition_size,
            'is_convex': is_convex,
            'violation_rate': violation_rate
        })
    return results


def run_convexity_grid_search(data_file, fpo_id="ALL", coalition_sizes=None, 
                             beta_range=None, theta_range=None,
                             commission=0.3, training_cost=50000, 
                             samples=50, cores=None, use_optimizer=False):
    """
    Performs a grid search (or optimization) over beta and theta parameters to find values 
    that ensure game convexity across different coalition sizes.
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    if fpo_id != "ALL":
        df = df[df['FPO_ID'] == fpo_id]
        print(f"Filtered to {len(df)} farmers in FPO {fpo_id}")
    
    if coalition_sizes is None:
        coalition_sizes = [3, 4, 5, 6, 7, 10, 15, 20, 25]
    
    if beta_range is None:
        beta_range = np.linspace(0.2, 0.4, 21)   # finer grid
    
    if theta_range is None:
        theta_range = np.linspace(0.7, 0.9, 21)

    # === Continuous optimization mode ===
    if use_optimizer:
        print("\nRunning optimizer to minimize convexity violations...")
        def convexity_loss(x, df, coalition_sizes, commission, training_cost, samples):
            beta, theta = x
            params = {
                'commission': commission,
                'beta': beta,
                'theta': theta,
                'training_cost': training_cost
            }
            total_viol = 0
            for coalition_size in coalition_sizes:
                sample_df = df.sample(n=min(coalition_size, len(df)), random_state=42)
                _, violation_rate = is_game_convex(sample_df, params, exponential_synergy_value, samples=samples)
                total_viol += violation_rate
            return total_viol

        res = minimize(
            convexity_loss, x0=[0.3, 0.8],
            args=(df, coalition_sizes, commission, training_cost, samples),
            bounds=[(0.01, 2.0), (0.5, 2.0)]
        )
        print("\n=== Optimizer Results ===")
        print(f"Optimal beta: {res.x[0]:.4f}, theta: {res.x[1]:.4f}, objective={res.fun:.4f}")
        return res

    # === Grid search mode ===
    print(f"\nRunning grid search with:")
    print(f"  - Coalition sizes: {coalition_sizes}")
    print(f"  - Beta values: {beta_range}")
    print(f"  - Theta values: {theta_range}")
    print(f"  - Total parameter combinations: {len(beta_range) * len(theta_range)}")
    print(f"  - Samples per combination: {samples}")
    
    param_grid = list(product(beta_range, theta_range))

    # Set up parallelization
    if cores is None:
        cores = mp.cpu_count()
    print(f"Using {cores} CPU cores for parallel processing")

    # Pre-sample farmers ONCE per coalition size to keep results consistent across β/θ
    farmer_samples = {
        size: df.sample(n=min(size, len(df)), random_state=42) for size in coalition_sizes
    }
    
    # Create a partial function with fixed parameters
    test_func = partial(
        test_params,
        farmer_samples=farmer_samples,
        commission=commission,
        training_cost=training_cost,
        samples=samples
    )

    print("\nStarting grid search (this may take a while)...")
    with mp.Pool(cores) as pool:
        all_results = list(tqdm(
            pool.imap(test_func, param_grid, chunksize=5),
            total=len(param_grid)
        ))

    flat_results = [item for sublist in all_results for item in sublist]
    results_df = pd.DataFrame(flat_results)

    # Summaries
    summary = results_df.groupby(['beta', 'theta']).agg({
        'is_convex': 'mean',
        'violation_rate': 'mean'
    }).reset_index()
    summary['convexity_score'] = summary['is_convex']

    print("\n=== Top Parameter Combinations for Convexity ===")
    print(summary.sort_values('convexity_score', ascending=False).head(10))

    # Check fully convex combos
    always_convex = []
    for beta, theta in param_grid:
        subset = results_df[(results_df['beta'] == beta) & (results_df['theta'] == theta)]
        if subset['is_convex'].all():
            always_convex.append({'beta': beta, 'theta': theta})

    if always_convex:
        always_convex_df = pd.DataFrame(always_convex)
        print("\n=== Parameter Combinations with 100% Convexity ===")
        print(always_convex_df)
    else:
        print("\nNo parameter combinations ensure convexity for all coalition sizes")

    # Save
    output_dir = './data/results/convexity_grid_search'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
    summary.to_csv(f'{output_dir}/summary.csv', index=False)

    # Heatmaps
    create_convexity_heatmaps(summary, coalition_sizes, output_dir)

    return results_df, summary, always_convex


def create_convexity_heatmaps(summary, coalition_sizes, output_dir):
    """Creates heatmaps for convexity probability and violation rates"""
    for metric, cmap, title in [
        ('convexity_score', 'YlGnBu', 'Convexity Probability'),
        ('violation_rate', 'Reds_r', 'Convexity Violation Rate')  # Notice Reds_r (reversed)
    ]:
        plt.figure(figsize=(10, 8))
        heatmap_data = summary.pivot(index='beta', columns='theta', values=metric)
        
        # For violation rate, lower is better, so we need to adjust vmin/vmax
        vmin = 0
        vmax = 1
        if metric == 'violation_rate':
            vmin = 0
            vmax = min(1, summary['violation_rate'].max() * 1.1)  # Scale to max observed
        
        ax = sns.heatmap(heatmap_data, annot=True, cmap=cmap, fmt='.2f', vmin=vmin, vmax=vmax)
        plt.title(f'{title} by Beta and Theta\nTested on Coalition Sizes: {coalition_sizes}', fontsize=14, pad=20)
        plt.xlabel('Theta (Synergy Exponent)', fontsize=12)
        plt.ylabel('Beta (Synergy Strength)', fontsize=12)
        plt.tight_layout()
        path = f'{output_dir}/{metric}_heatmap.png'
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved {title} heatmap to {path}")


if __name__ == "__main__":
    results, summary, always_convex = run_convexity_grid_search(
        data_file="./data/synthetic/farmer_data_v3.csv",
        fpo_id="FPO_B",
        coalition_sizes=[2, 3, 4, 5, 6, 7, 10, 12, 15, 17,  20,  25,28,30,33,36,37,40],
        beta_range=np.linspace(0.2, 0.4, 21),
        theta_range=np.linspace(0.7, 0.9, 21),
        commission=0.3,
        training_cost=50000,
        samples=100,  # Reduced from 500 to run faster
        use_optimizer=False  # set True to run continuous optimization instead
    )