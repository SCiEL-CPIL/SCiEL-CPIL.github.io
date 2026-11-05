import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check for zero weight items with positive value
    zero_weight_mask = (remaining_weights == 0) & (remaining_values > 0)
    if np.any(zero_weight_mask):
        return int(np.where(zero_weight_mask)[0][0])

    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    m = len(feasible_indices)
    
    # Calculate densities
    densities = feasible_values / feasible_weights
    
    # Sort indices by density descending
    sorted_order = np.argsort(-densities)
    
    # Build candidate pool: top by density + top by value
    top_k = min(m, max(10, m // 5))
    top_density = set(sorted_order[:top_k].tolist())
    top_value_order = np.argsort(-feasible_values)
    top_value = set(top_value_order[:top_k].tolist())
    
    # Union of candidates
    candidate_set = top_density | top_value
    if m <= 50:
        candidate_set = set(range(m))
    
    candidates = sorted(candidate_set)
    
    # Pre-extract sorted arrays for faster access
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    
    best_score = -1
    best_idx = -1
    start_time = time.time()
    deadline = start_time + 0.8  # Safety margin per call
    
    for k in candidates:
        if time.time() > deadline:
            break
            
        idx_in_feasible = feasible_indices[k]
        w = feasible_weights[k]
        v = feasible_values[k]
        
        # Simulate picking this item
        rem_cap = remaining_capacity - w
        sim_val = v
        
        # Greedy fill the rest using pre-sorted arrays
        for j in range(m):
            if j == k:
                continue
            w_j = sorted_weights[j]
            if w_j <= rem_cap:
                sim_val += sorted_values[j]
                rem_cap -= w_j
                if rem_cap <= 0:
                    break
        
        if sim_val > best_score:
            best_score = sim_val
            best_idx = idx_in_feasible

    if best_idx == -1:
        best_idx = feasible_indices[sorted_order[0]]
        
    return int(best_idx)
