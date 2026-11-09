"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time budget
    if not _elapsed_ok():
        # Fallback to simple greedy if time is tight
        densities = remaining_values / remaining_weights
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        # Mask out infeasible
        masked_densities = np.where(feasible, densities, -1)
        return int(np.argmax(masked_densities))

    # 1. Compute densities for all remaining items
    densities = remaining_values / remaining_weights
    
    # 2. Sort indices by density descending
    # argsort with negative sign to get descending order
    sorted_indices = np.argsort(-densities)
    
    best_total_value = -1.0
    best_index = -1
    
    # 3. Evaluate each feasible item as the next pick
    # We iterate through all items, but only consider those that fit
    for i in range(n):
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
        
        # If we pick item i, we have remaining_capacity - w_i left
        # and we want to fill it with the best density items from the rest
        sim_cap = remaining_capacity - w_i
        sim_val = remaining_values[i]
        
        # Iterate through sorted items to greedily fill the rest
        # We skip item i itself
        for idx in sorted_indices:
            if idx == i:
                continue
            w_idx = remaining_weights[idx]
            if w_idx <= sim_cap:
                sim_cap -= w_idx
                sim_val += remaining_values[idx]
                if sim_cap <= 0:
                    break
        
        if sim_val > best_total_value:
            best_total_value = sim_val
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
