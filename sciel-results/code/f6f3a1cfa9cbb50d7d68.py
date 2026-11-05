import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _greedy_fill_value(capacity, weights, values, exclude_idx):
    """
    Estimate the value of filling the knapsack with remaining items
    using a greedy density strategy, excluding item at exclude_idx.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create indices of all items except exclude_idx
    # We need to be careful with numpy operations for speed
    idxs = np.arange(n)
    if exclude_idx >= 0:
        idxs = np.delete(idxs, exclude_idx)
    
    if len(idxs) == 0:
        return 0.0
        
    w_rem = weights[idxs]
    v_rem = values[idxs]
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(w_rem > 0, v_rem / w_rem, np.inf)
        
    # Sort by density descending
    # argsort is stable by default? We want highest density first.
    # np.argsort sorts ascending. So we take reverse.
    order = np.argsort(densities)[::-1]
    
    total_val = 0.0
    rem_cap = capacity
    
    for idx in order:
        w = w_rem[idx]
        v = v_rem[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
            if rem_cap <= 0:
                break
        else:
            # Since we are in 0/1 knapsack, we skip if it doesn't fit.
            # Greedy density usually just skips. 
            # Could we do better? Maybe check if a smaller item later fits?
            # The loop continues, so it will check subsequent items.
            pass
            
    return total_val

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
    
    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    best_val = -1.0
    best_idx = -1
    
    # Evaluate each feasible item as the next pick
    # Optimization: If we have many items, we might want to limit candidates to top K by density
    # to save time, but N=100 is small. Let's evaluate all feasible.
    
    # Sort feasible indices by density descending to potentially find good candidates early?
    # Doesn't matter for correctness, but might help if we add early termination.
    # For now, just loop.
    
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        
        # Check time again inside loop if N was large, but N=100 is fine.
        # if time.time() - _start_time > TIME_LIMIT:
        #     break
        
        new_cap = remaining_capacity - w_i
        fill_val = _greedy_fill_value(new_cap, weights, values, i)
        total_est = v_i + fill_val
        
        if total_est > best_val:
            best_val = total_est
            best_idx = i
            
    return int(best_idx)
