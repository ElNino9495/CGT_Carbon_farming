import numpy as np
import pandas as pd
import config
from characteristic_functions import characteristic_function_v
import itertools
import math
import warnings

def monte_carlo_shapley(coalition_df, params, v_func, samples=None):
    
    if samples is None:
        samples = config.SHAPLEY_SAMPLES
        
    n_players = len(coalition_df)
    if n_players == 0:
        return {}
    
    player_ids = coalition_df['Farmer_ID'].tolist()
    shapley_values = {pid: 0.0 for pid in player_ids}
    
    player_data = {
        row['Farmer_ID']: row 
        for _, row in coalition_df.iterrows()
    }

    for _ in range(samples):
        np.random.shuffle(player_ids)
        
        current_coalition_ids = []
        v_previous = 0.0
        
        for player_id in player_ids:
            current_coalition_ids.append(player_id)
            
            current_coalition_rows = [player_data[pid] for pid in current_coalition_ids]
            current_df = pd.DataFrame(current_coalition_rows)
            
            v_current = v_func(current_df, params)
            
            marginal_contribution = v_current - v_previous
            
            shapley_values[player_id] += marginal_contribution
            
            v_previous = v_current
            
    for player_id in shapley_values:
        shapley_values[player_id] /= samples
        
    return shapley_values


def exact_shapley_values(coalition_df, params, v_func):
   
    n_players = len(coalition_df)
    if n_players == 0:
        return {}
        
 
    player_ids = coalition_df['Farmer_ID'].tolist()
    shapley_values = {pid: 0.0 for pid in player_ids}
    
    player_data = {
        row['Farmer_ID']: row 
        for _, row in coalition_df.iterrows()
    }
    
    precomputed_values = {}
    for k in range(n_players + 1):
        for subset_ids_tuple in itertools.combinations(player_ids, k):
            subset_ids = frozenset(subset_ids_tuple)
            
            if not subset_ids:
                precomputed_values[subset_ids] = 0.0
                continue
                
            subset_rows = [player_data[pid] for pid in subset_ids]
            subset_df = pd.DataFrame(subset_rows)
            
            precomputed_values[subset_ids] = v_func(subset_df, params)
            
    for player_id in player_ids:
        phi_i = 0.0
        
        for subset_ids in precomputed_values:
            if player_id not in subset_ids:
                s = len(subset_ids)
                
                v_S = precomputed_values[subset_ids]
                
                S_union_i = subset_ids.union({player_id})
                v_S_union_i = precomputed_values[S_union_i]
                
                marginal_contrib = v_S_union_i - v_S
                
                weight = (math.factorial(s) * math.factorial(n_players - s - 1)) / math.factorial(n_players)
                
                phi_i += weight * marginal_contrib
                
        shapley_values[player_id] = phi_i
        
    return shapley_values