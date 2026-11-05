"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item using a lookahead greedy strategy.
    For each feasible item, simulates the greedy completion of the knapsack
    and picks the item that yields the highest total value.
    Falls back to simple density greedy if time is running out.
    """
    current_time = time.time()
    if current_time - _start_time > _TIME_LIMIT:
        # Fallback: simple density greedy
        if len(remaining_weights) == 0:
            return -1
        densities = remaining_values / remaining_weights
        return int(np.argmax(densities))

    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Pre-sort all items by density descending for simulation
    densities = remaining_values / remaining_weights
    sorted_indices = np.argsort(-densities)
    
    best_value = -1
    best_idx = -1
    
    # Evaluate each feasible candidate
    for cand in feasible_indices:
        # Start with candidate
        cur_cap = remaining_capacity - remaining_weights[cand]
        total_val = remaining_values[cand]
        
        # Simulate greedy fill with remaining items
        for idx in sorted_indices:
            if idx == cand:
                continue
            w = remaining_weights[idx]
            if w <= cur_cap:
                cur_cap -= w
                total_val += remaining_values[idx]
            # Optimization: if cur_cap is very small, we could break early?
            # No, a small weight item might still fit.
        
        if total_val > best_value:
            best_value = total_val
            best_idx = cand
            
    return int(best_idx)
# EVOLVE-BLOCK-END
