import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _greedy_fill_value(capacity, weights, values, exclude_idx):
    """
    Estimate the value of filling the knapsack with remaining items
    using a greedy density strategy with best-fit refinement, excluding item at exclude_idx.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create indices of all items except exclude_idx
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
    order = np.argsort(densities)[::-1]
    
    total_val = 0.0
    rem_cap = capacity
    used = set()
    
    # First pass: greedy by density
    for idx in order:
        if idx in used:
            continue
        w = w_rem[idx]
        v = v_rem[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
            used.add(idx)
            if rem_cap <= 0:
                break
    
    # Second pass: try to improve by swapping low-density items with better fits
    # This is a simple local improvement
    if rem_cap > 0:
        # Find items that were skipped but might fit better
        for idx in order:
            if idx in used:
                continue
            w = w_rem[idx]
            v = v_rem[idx]
            if w <= rem_cap:
                # Check if we can swap with a low-density used item
                best_swap_val = 0
                best_swap_idx = -1
                for used_idx in used:
                    if w_rem[used_idx] >= w and v_rem[used_idx] < v:
                        # Swapping would improve value
                        gain = v - v_rem[used_idx]
                        if gain > best_swap_val:
                            best_swap_val = gain
                            best_swap_idx = used_idx
                if best_swap_idx >= 0:
                    total_val += best_swap_val
                    rem_cap += w_rem[best_swap_idx] - w
                    used.remove(best_swap_idx)
                    used.add(idx)
                    if rem_cap <= 0:
                        break
    
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
    
    # Pre-compute densities for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        all_densities = np.where(weights > 0, values / weights, np.inf)
    
    # Sort feasible indices by density descending for potential early termination
    feasible_sorted = feasible_indices[np.argsort(all_densities[feasible_indices])[::-1]]
    
    for i in feasible_sorted:
        w_i = weights[i]
        v_i = values[i]
        
        # Check time again inside loop if N was large, but N=100 is fine.
        if time.time() - _start_time > TIME_LIMIT:
            break
        
        new_cap = remaining_capacity - w_i
        fill_val = _greedy_fill_value(new_cap, weights, values, i)
        total_est = v_i + fill_val
        
        if total_est > best_val:
            best_val = total_est
            best_idx = i
            
    return int(best_idx)
