import numpy as np
import time

def _simulate_greedy_value_fast(capacity, weights, values, sorted_indices, exclude_set):
    """
    Simulates greedy packing using a pre-sorted list of indices by density.
    Skips items in exclude_set.
    """
    total_value = 0.0
    current_cap = capacity
    
    for idx in sorted_indices:
        if idx in exclude_set:
            continue
        w = weights[idx]
        if w > current_cap:
            continue
        total_value += values[idx]
        current_cap -= w
        if current_cap <= 1e-9:
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0 # 100s margin
    
    # Pre-compute densities and sort indices
    # We want descending density
    densities = np.zeros(n)
    for i in range(n):
        if remaining_weights[i] > 0:
            densities[i] = remaining_values[i] / remaining_weights[i]
        else:
            densities[i] = float('inf') if remaining_values[i] > 0 else 0.0
            
    sorted_indices = np.argsort(-densities) # Descending order
    
    best_total_val = -1.0
    best_idx = -1
    
    # 2-step lookahead
    # For each i, find the best j (including "none") to maximize total value
    for i in range(n):
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = remaining_values[i]
        
        # Option 1: Pick i, then stop (or let greedy handle the rest without a specific second choice constraint)
        # Actually, the "greedy remainder" in the 1-step version already picks the best next items.
        # The 2-step version allows us to "reserve" a specific good second item j that might be
        # overlooked by pure greedy if a slightly less dense but heavier item blocks it?
        # No, greedy picks best density that fits.
        # The improvement comes from considering that picking i might block a very high value item j
        # that is slightly less dense than another item k, but k doesn't fit well with the rest.
        # By explicitly evaluating pairs (i, j), we capture interactions.
        
        best_score_for_i = -1.0
        
        # Try all j as the second item
        for j in range(n):
            if j == i:
                continue
            w_j = remaining_weights[j]
            if w_i + w_j > remaining_capacity:
                continue
                
            v_j = remaining_values[j]
            
            # Simulate greedy for the rest, excluding i and j
            exclude_set = {i, j}
            future_val = _simulate_greedy_value_fast(
                remaining_capacity - w_i - w_j, 
                remaining_weights, 
                remaining_values, 
                sorted_indices, 
                exclude_set
            )
            
            total_val = v_i + v_j + future_val
            if total_val > best_score_for_i:
                best_score_for_i = total_val
        
        # Also consider the case where we pick i, and the *best* next item is determined by greedy
        # (This is covered by the j-loop if j is the greedy choice, but what if no j fits?
        # If no j fits, the loop doesn't run. We must handle the "only i" case.)
        if best_score_for_i == -1.0:
            # No second item fits. Just pick i and let greedy handle the rest (which will find 0 more)
            # Or, simulate greedy for the rest excluding i
            exclude_set = {i}
            future_val = _simulate_greedy_value_fast(
                remaining_capacity - w_i, 
                remaining_weights, 
                remaining_values, 
                sorted_indices, 
                exclude_set
            )
            best_score_for_i = v_i + future_val

        if best_score_for_i > best_total_val:
            best_total_val = best_score_for_i
            best_idx = i
            
        # Check time
        if time.time() > deadline:
            break
            
    return int(best_idx)
