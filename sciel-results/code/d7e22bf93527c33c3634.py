import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 145.0

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    if time.time() - _start_time > _TIMEOUT:
        return -1

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    if C < 0:
        return -1
        
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    n_feas = len(feasible_indices)
    
    # Handle zero-weight items: pick them all if they have positive value
    zero_w_pos_v = (w == 0) & (v > 0)
    if np.any(zero_w_pos_v):
        # Return the first zero-weight positive-value item
        return int(np.where(zero_w_pos_v)[0][0])
    
    # For positive weight feasible items, compute a better score
    # Score = v_c + fractional_knapsack_upper_bound(C - w_c, remaining items except c)
    
    # Precompute density for all items
    pos_w = w > 0
    densities = np.zeros(n)
    densities[pos_w] = v[pos_w] / w[pos_w]
    
    # Sort all feasible items by density descending for fractional knapsack simulation
    # We'll use the sorted order to quickly compute fractional upper bound
    
    # For each candidate c in feasible_indices:
    #   sim_val = v[c] + fractional_bound(C - w[c], items excluding c)
    
    # To speed up, sort feasible items by density once
    feas_w = w[feasible_indices]
    feas_v = v[feasible_indices]
    feas_dens = densities[feasible_indices]
    
    # Sort by density descending
    sort_order = np.argsort(-feas_dens, kind='stable')
    sorted_feas_w = feas_w[sort_order]
    sorted_feas_v = feas_v[sort_order]
    sorted_feas_idx = feasible_indices[sort_order]
    
    # Precompute prefix sums for fast fractional knapsack
    # prefix_w[i] = sum of sorted_feas_w[0..i-1]
    # prefix_v[i] = sum of sorted_feas_v[0..i-1]
    prefix_w = np.zeros(n_feas + 1)
    prefix_v = np.zeros(n_feas + 1)
    for i in range(n_feas):
        prefix_w[i+1] = prefix_w[i] + sorted_feas_w[i]
        prefix_v[i+1] = prefix_v[i] + sorted_feas_v[i]
    
    # For each candidate, compute the fractional knapsack upper bound
    # excluding that candidate
    best_score = -1e10
    best_idx = -1
    
    # Map from original feasible index to position in sorted order
    pos_in_sorted = np.full(n_feas, -1)
    for i in range(n_feas):
        pos_in_sorted[sort_order[i]] = i
    
    for ci in range(n_feas):
        c = feasible_indices[ci]
        w_c = w[c]
        v_c = v[c]
        rem_cap = C - w_c
        
        if rem_cap < 0:
            continue
        
        # Compute fractional knapsack upper bound on remaining capacity,
        # excluding item c
        # Item c is at position pos_in_sorted[ci] in the sorted array
        pos_c = pos_in_sorted[ci]
        
        # We need to fill rem_cap using all sorted items except the one at pos_c
        # Iterate through sorted items, skipping pos_c
        
        sim_val = v_c
        cap = rem_cap
        
        for i in range(n_feas):
            if i == pos_c:
                continue
            w_i = sorted_feas_w[i]
            v_i = sorted_feas_v[i]
            if cap <= 0:
                break
            if w_i <= cap:
                cap -= w_i
                sim_val += v_i
            else:
                # Fractional: take cap/w_i fraction
                if w_i > 0:
                    sim_val += v_i * (cap / w_i)
                cap = 0
                break
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = c
    
    if best_idx < 0 or best_score <= 0:
        return -1
        
    return int(best_idx)
