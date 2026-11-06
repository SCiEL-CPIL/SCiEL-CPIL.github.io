"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    if _START_TIME is None:
        return True
    # Safety margin: 115s out of 120s limit (or 150s if evaluator uses that)
    # The prompt says "KILLS ... after 150s". Let's use 140s to be safe.
    return (time.time() - _START_TIME) < 140.0

def _simulate_greedy_fill(remaining_capacity, remaining_weights, remaining_values, sorted_indices, exclude_index):
    """
    Simulates the total value obtained by greedily filling the knapsack
    starting with the remaining capacity, excluding the item at exclude_index.
    """
    if remaining_capacity <= 0:
        return 0.0
    
    total_val = 0.0
    cap = remaining_capacity
    
    # Iterate through items sorted by density
    for idx in sorted_indices:
        if idx == exclude_index:
            continue
        w = remaining_weights[idx]
        if w <= cap:
            total_val += remaining_values[idx]
            cap -= w
            if cap <= 0:
                break
    
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Check time
    if not _check_time():
        # Fallback to simple greedy if time is tight
        densities = np.zeros(n)
        for i in range(n):
            if remaining_weights[i] > 0:
                densities[i] = remaining_values[i] / remaining_weights[i]
            else:
                densities[i] = np.inf
        # Only consider feasible items
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        # Find max density among feasible
        best_idx = -1
        best_dens = -1
        for i in feasible:
            if densities[i] > best_dens:
                best_dens = densities[i]
                best_idx = i
        return int(best_idx)

    # Identify feasible items
    feasible_indices = []
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity:
            feasible_indices.append(i)
    
    if not feasible_indices:
        return -1

    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Sort all indices by density (value/weight) descending
    # We need to handle zero weights (infinity density)
    densities = np.zeros(n)
    for i in range(n):
        if remaining_weights[i] > 0:
            densities[i] = remaining_values[i] / remaining_weights[i]
        else:
            densities[i] = np.inf
            
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_item = -1
    best_total_value = -1.0

    # Lookahead: For each feasible item, simulate the rest of the greedy process
    for i in feasible_indices:
        new_cap = remaining_capacity - remaining_weights[i]
        # Value of item i + Value of greedy fill from remaining items
        sim_val = remaining_values[i] + _simulate_greedy_fill(
            new_cap, 
            remaining_weights, 
            remaining_values, 
            sorted_indices, 
            i
        )
        
        if sim_val > best_total_value:
            best_total_value = sim_val
            best_item = i

    return int(best_item)
# EVOLVE-BLOCK-END
