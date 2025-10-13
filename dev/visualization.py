import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def visualize_growth_trends(results_df, output_dir):
  
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fpo_order = sorted(results_df['fpo_id'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Game Properties vs. Coalition Size', fontsize=18, weight='bold')

    sns.lineplot(
        data=results_df, 
        x='num_farmers', 
        y='core_stable', 
        hue='fpo_id',
        hue_order=fpo_order,
        ax=axes[0],
        errorbar='sd'
    )
    axes[0].set_title('Likelihood of a Stable Core', fontsize=14)
    axes[0].set_ylabel('Probability of Stable Core')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].get_legend().set_title('FPO ID')

    sns.lineplot(
        data=results_df, 
        x='num_farmers', 
        y='is_convex', 
        hue='fpo_id',
        hue_order=fpo_order,
        ax=axes[1],
        errorbar='sd'
    )
    axes[1].set_title('Likelihood of a Convex Game', fontsize=14)
    axes[1].set_ylabel('Probability of Convexity')
    axes[1].set_xlabel('Number of Farmers in Coalition', fontsize=12)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].get_legend().set_title('FPO ID')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/1_stability_vs_size.png', dpi=300)
    plt.close(fig)
    print(f"Saved stability trend plot to {output_dir}/1_stability_vs_size.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.lineplot(
        data=results_df,
        x='num_farmers',
        y='avg_shapley',
        hue='fpo_id',
        hue_order=fpo_order,
        ax=ax,
        errorbar='sd'
    )
    
    ax.set_title('Average Farmer Payoff (Shapley Value) vs. Coalition Size', fontsize=16, weight='bold')
    ax.set_ylabel('Average Payoff per Farmer (INR)', fontsize=12)
    ax.set_xlabel('Number of Farmers in Coalition', fontsize=12)
    ax.axhline(0, color='grey', linestyle='--', lw=1)
    ax.get_legend().set_title('FPO ID')

    fig.tight_layout()
    plt.savefig(f'{output_dir}/2_payoff_vs_size.png', dpi=300)
    plt.close(fig)
    print(f"Saved payoff trend plot to {output_dir}/2_payoff_vs_size.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.lineplot(
        data=results_df,
        x='num_farmers',
        y='total_project_profit', 
        hue='fpo_id',
        hue_order=fpo_order,
        ax=ax,
        errorbar='sd'
    )
    
    ax.set_title('Total Project Profit (Group Revenue) vs. Coalition Size', fontsize=16, weight='bold')
    ax.set_ylabel('Total Project Profit (INR)', fontsize=12)
    ax.set_xlabel('Number of Farmers in Coalition', fontsize=12)
    ax.axhline(0, color='grey', linestyle='--', lw=1)
    ax.get_legend().set_title('FPO ID')
    
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


    fig.tight_layout()
    plt.savefig(f'{output_dir}/3_group_profit_vs_size.png', dpi=300)
    plt.close(fig)
    print(f"Saved group profit trend plot to {output_dir}/3_group_profit_vs_size.png")