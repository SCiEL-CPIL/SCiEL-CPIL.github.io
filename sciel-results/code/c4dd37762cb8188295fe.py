import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _fractional_fill_value(capacity, weights, values, exclude_idx):
    """
    Estimate the value of filling the knapsack with remaining items
    using the Fractional Knapsack (LP relaxation) bound.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    # Create indices of all items except exclude_idx
    idxs = np.arange(n)
    if exclude_idx >= 0:
        idxs = idxs[idxs != exclude_idx]
        
    if len(idxs) == 0:
        return 0.0
        
    w_rem = weights[idxs]
    v_rem = values[idxs]
    
    # Filter feasible items for this sub-problem (weight > 0)
    # In fractional knapsack, we can take any fraction, so weight > 0 is the only constraint
    # besides capacity which is handled in the loop.
    pos_mask = w_rem > 0
    w_f = w_rem[pos_mask]
    v_f = v_rem[pos_mask]
    
    if len(w_f) == 0:
        return 0.0
        
    # Calculate densities
    densities = v_f / w_f
    
    # Sort by density descending
    order = np.argsort(densities)[::-1]
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        if remaining_cap <= 0:
            break
        w = w_f[idx]
        v = v_f[idx]
        
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
        else:
            # Take fraction
            total_value += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Time check
    if time.time() - _start_time > TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Identify feasible items (weight <= remaining_capacity)
    # Note: For 0/1 knapsack, an item must fit entirely to be selected.
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    best_val = -1.0
    best_idx = -1
    
    # Evaluate each feasible item as the next pick
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        
        # Check time again inside loop if N was large, but N=100 is fine.
        # if time.time() - _start_time > TIME_LIMIT:
        #     break
        
        new_cap = remaining_capacity - w_i
        fill_val = _fractional_fill_value(new_cap, weights, values, i)
        total_est = v_i + fill_val
        
        if total_est > best_val:
            best_val = total_est
            best_idx = i
            
    return int(best_idx)
