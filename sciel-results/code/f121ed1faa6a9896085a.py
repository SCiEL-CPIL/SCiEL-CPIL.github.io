import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # 1. Identify feasible items
    # We must ensure we only pick items that fit
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # 2. Calculate densities for feasible items
    # Avoid division by zero, though weights should be > 0
    # If weight is 0, density is inf. 
    # Problem statement implies standard knapsack, weights > 0.
    # But to be safe:
    weights_f = remaining_weights[feasible_indices]
    values_f = remaining_values[feasible_indices]
    
    # Handle zero weights if any (unlikely in knapsack, but safe)
    # If w=0, we should definitely pick it if v>0.
    # Density approach breaks with div by zero.
    # Let's assume weights > 0. If not, handle separately.
    
    if np.any(weights_f == 0):
        # Pick the zero-weight item with highest value
        zero_idx_local = np.where(weights_f == 0)[0]
        if len(zero_idx_local) > 0:
            best_zero_local = zero_idx_local[np.argmax(values_f[zero_idx_local])]
            return int(feasible_indices[best_zero_local])
        # If zero weight items have 0 value, ignore them for density calc
        # Filter them out for the rest of the logic?
        # Actually, if w=0 and v=0, they don't help.
        # If w=0 and v>0, we picked the best one.
        # Now we need to pick the next one.
        # The logic below assumes w>0.
        # Let's just filter out w=0 from the main density calc if we haven't returned.
        # But if we returned, we are done.
        # If we didn't return, it means no w=0 items or all had v=0?
        # If w=0 and v=0, picking them doesn't change capacity or value.
        # We can ignore them.
        non_zero_mask = weights_f > 0
        if not np.any(non_zero_mask):
             return -1 # Only zero value zero weight items left? Or just zero weights.
        feasible_indices = feasible_indices[non_zero_mask]
        weights_f = weights_f[non_zero_mask]
        values_f = values_f[non_zero_mask]
        
    densities_f = values_f / weights_f
    
    # 3. Find max and second max density
    # We need to know the max density and the second max density among feasible items
    # To determine max_d_rest for each item
    
    # Sort indices by density descending
    sorted_indices = np.argsort(densities_f)[::-1]
    
    max_d = densities_f[sorted_indices[0]]
    if len(sorted_indices) > 1:
        second_max_d = densities_f[sorted_indices[1]]
    else:
        second_max_d = 0.0
        
    max_d_idx = sorted_indices[0]
    
    # 4. Score each feasible item
    scores = np.zeros(len(feasible_indices))
    
    for i in range(len(feasible_indices)):
        w_i = weights_f[i]
        v_i = values_f[i]
        
        rem_cap = remaining_capacity - w_i
        
        # Determine max density of remaining items (excluding i)
        if i == max_d_idx:
            # This item holds the max density
            # Use second max
            max_d_rest = second_max_d
        else:
            # Max density is held by someone else
            max_d_rest = max_d
            
        future_value = rem_cap * max_d_rest
        scores[i] = v_i + future_value
        
    # 5. Pick best
    best_local_idx = np.argmax(scores)
    return int(feasible_indices[best_local_idx])
