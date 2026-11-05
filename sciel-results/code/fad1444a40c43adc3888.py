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

def _dp_fill_value(capacity, weights, values, scale=1000):
    """
    Compute the exact 0/1 knapsack value using DP with discretized capacity.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    cap_units = int(capacity * scale)
    if cap_units <= 0:
        return 0.0
    
    w_units = np.round(weights * scale).astype(int)
    feasible = (w_units > 0) & (w_units <= cap_units)
    
    if not np.any(feasible):
        return 0.0
    
    dp = np.zeros(cap_units + 1, dtype=float)
    for i in range(len(weights)):
        if feasible[i]:
            w = w_units[i]
            v = values[i]
            dp[w:] = np.maximum(dp[w:], dp[:-w] + v)
    
    return dp[cap_units]

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
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    best_val = -1.0
    best_idx = -1
    
    # Sort feasible indices by value/weight ratio for better early termination
    ratios = np.zeros(len(feasible_indices))
    for k, i in enumerate(feasible_indices):
        if weights[i] > 0:
            ratios[k] = values[i] / weights[i]
        else:
            ratios[k] = float('inf')
    
    order = np.argsort(ratios)[::-1]
    sorted_indices = feasible_indices[order]
    
    # Evaluate each feasible item using DP-based bound
    for i in sorted_indices:
        if time.time() - _start_time > TIME_LIMIT - 10.0:
            break
            
        w_i = weights[i]
        v_i = values[i]
        
        new_cap = remaining_capacity - w_i
        
        if new_cap > 0:
            idxs = np.arange(n)
            idxs = idxs[idxs != i]
            w_rem = weights[idxs]
            v_rem = values[idxs]
            
            fill_val = _dp_fill_value(new_cap, w_rem, v_rem)
        else:
            fill_val = 0.0
        
        total_est = v_i + fill_val
        
        if total_est > best_val:
            best_val = total_est
            best_idx = i
    
    if best_idx == -1:
        # Fallback: pick the feasible item with highest value/weight ratio
        best_idx = int(sorted_indices[0])
    
    return int(best_idx)
