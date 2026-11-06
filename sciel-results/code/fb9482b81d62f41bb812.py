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
    # Safety margin: 145s out of 150s limit
    return (time.time() - _START_TIME) < 145.0

def _simulate_greedy_fill(remaining_capacity, remaining_weights, remaining_values, sorted_indices, exclude_indices):
    """
    Simulates the total value obtained by greedily filling the knapsack
    starting with the remaining capacity, excluding items in exclude_indices.
    """
    if remaining_capacity <= 0:
        return 0.0
    
    total_val = 0.0
    cap = remaining_capacity
    
    # Extract excluded indices for fast comparison
    ex1 = -1
    ex2 = -1
    if len(exclude_indices) > 0:
        ex1 = exclude_indices[0]
    if len(exclude_indices) > 1:
        ex2 = exclude_indices[1]

    # Iterate through items sorted by density
    for idx in sorted_indices:
        if idx == ex1 or idx == ex2:
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
    
    # Phase 1: 1-step lookahead to find top K candidates
    scores_1 = {}
    for i in feasible_indices:
        new_cap = remaining_capacity - remaining_weights[i]
        sim_val = remaining_values[i] + _simulate_greedy_fill(
            new_cap, 
            remaining_weights, 
            remaining_values, 
            sorted_indices, 
            [i]
        )
        scores_1[i] = sim_val
        
    # Determine K for 2-step lookahead
    if n <= 30:
        K = len(feasible_indices)
    else:
        K = min(20, len(feasible_indices))
        
    # Sort indices by score_1 descending
    sorted_candidates = sorted(scores_1.keys(), key=lambda x: scores_1[x], reverse=True)
    top_candidates = sorted_candidates[:K]
    
    # Phase 2: 2-step lookahead for top candidates
    best_item = -1
    best_total_value = -1.0
    
    for i in top_candidates:
        cap_after_i = remaining_capacity - remaining_weights[i]
        val_i = remaining_values[i]
        
        # Find feasible j for this i
        for j in feasible_indices:
            if j == i:
                continue
            if remaining_weights[j] > cap_after_i:
                continue
            
            new_cap = cap_after_i - remaining_weights[j]
            val_j = remaining_values[j]
            
            # Check time before doing the expensive simulation
            if not _check_time():
                # If time is tight, fallback to best 1-step candidate
                return int(sorted_candidates[0])
            
            sim_val = val_i + val_j + _simulate_greedy_fill(
                new_cap, 
                remaining_weights, 
                remaining_values, 
                sorted_indices, 
                [i, j]
            )
            
            if sim_val > best_total_value:
                best_total_value = sim_val
                best_item = i
                
    # If no 2-step improvement was found or loop didn't run
    if best_item == -1:
        best_item = sorted_candidates[0]
        
    return int(best_item)
# EVOLVE-BLOCK-END
