"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item using a 1-step lookahead heuristic.
    Considers top K items by density and simulates the greedy fill for each.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Check for any feasible items
    # If no item fits, return -1
    min_w = np.min(remaining_weights)
    if min_w > remaining_capacity:
        return -1

    # Calculate densities
    # Avoid division by zero, though weights should be positive
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = remaining_values / remaining_weights
    
    # Replace inf/nan with 0 for sorting stability if any weights are 0 (unlikely in knapsack)
    densities = np.where(np.isfinite(densities), densities, 0.0)

    # Get indices of top K candidates by density
    # K=50 is a good balance between exploration and speed for N=100
    K = min(50, n)
    # argsort in descending order
    sorted_indices = np.argsort(densities)[::-1]
    top_candidates = sorted_indices[:K]

    best_value = -1.0
    best_index = -1

    # Pre-sort all items by density once to use in simulation
    # We need to simulate: Pick candidate i, then greedily pick from others
    # To speed up simulation, we can have a master sorted list of all items
    # But we must exclude the candidate.
    
    # Let's just iterate. N is small (100).
    # For each candidate in top_candidates:
    for cand_idx in top_candidates:
        if not _elapsed_ok():
            break
            
        w_cand = remaining_weights[cand_idx]
        if w_cand > remaining_capacity:
            continue
            
        sim_cap = remaining_capacity - w_cand
        sim_val = remaining_values[cand_idx]
        
        # Simulate greedy fill with remaining items
        # We iterate through items sorted by density, skipping cand_idx
        for idx in sorted_indices:
            if idx == cand_idx:
                continue
            w_i = remaining_weights[idx]
            if w_i <= sim_cap:
                sim_cap -= w_i
                sim_val += remaining_values[idx]
            if sim_cap <= 0:
                break
        
        if sim_val > best_value:
            best_value = sim_val
            best_index = int(cand_idx)

    if best_index == -1:
        # Fallback to pure greedy if simulation failed or found nothing better
        # This shouldn't happen if there's a feasible item, but just in case
        return int(sorted_indices[0])
        
    return best_index
# EVOLVE-BLOCK-END
