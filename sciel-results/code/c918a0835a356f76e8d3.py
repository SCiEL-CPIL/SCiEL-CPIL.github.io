"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s

def _simulate_greedy_excluding(remaining_capacity, remaining_weights, remaining_values, 
                                sorted_indices, exclude_idx):
    """
    Simulates the standard greedy algorithm (by pre-sorted density) 
    on the remaining items, excluding 'exclude_idx'.
    Returns the total value achievable.
    """
    cap = remaining_capacity
    total_val = 0.0
    
    # We iterate through the pre-sorted indices
    for idx in sorted_indices:
        if idx == exclude_idx:
            continue
        w = remaining_weights[idx]
        if w <= cap:
            cap -= w
            total_val += remaining_values[idx]
            if cap <= 0:
                break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to simple greedy or stop
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # 1. Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # 2. Sort feasible items by value/weight density (descending)
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    densities = feasible_values / feasible_weights
    # Get local indices of feasible items sorted by density
    sorted_local_indices = np.argsort(densities)[::-1]
    
    # Map back to global indices
    sorted_global_indices = feasible_indices[sorted_local_indices]

    # 3. Determine K (number of candidates to look ahead)
    # If N is small, look at all. If large, look at top 30.
    # Increased from 20 to 30 based on inspiration 2's vectorization insight
    # (though we keep the loop, the logic is similar)
    K = min(len(sorted_global_indices), 30)
    
    best_candidate_idx = -1
    best_total_value = -1.0
    
    # 4. For each of the top K candidates, simulate the rest
    for i in range(K):
        candidate_idx = sorted_global_indices[i]
        
        # Value if we pick this candidate
        val_c = remaining_values[candidate_idx]
        w_c = remaining_weights[candidate_idx]
        
        # Remaining capacity after picking candidate
        cap_rem = remaining_capacity - w_c
        
        # Simulate greedy completion with remaining items
        sim_val = _simulate_greedy_excluding(cap_rem, remaining_weights, remaining_values, 
                                             sorted_global_indices, candidate_idx)
        
        total_val = val_c + sim_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_candidate_idx = candidate_idx

    return int(best_candidate_idx)
# EVOLVE-BLOCK-END
