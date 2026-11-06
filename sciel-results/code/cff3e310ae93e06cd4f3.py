"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s margin before 150s limit


def _simulate_greedy_completion(remaining_capacity, sorted_weights, sorted_values, skip_index=-1):
    """Simulate greedy packing by density, optionally skipping one item."""
    cap = remaining_capacity
    total_val = 0.0
    for j in range(len(sorted_weights)):
        if j == skip_index:
            continue
        item_w = sorted_weights[j]
        if item_w <= cap:
            cap -= item_w
            total_val += sorted_values[j]
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick next item or -1 to stop.
    Uses a lookahead heuristic that considers future capacity impact.
    """
    if time.time() > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = remaining_weights[feasible_mask]
    feasible_values = remaining_values[feasible_mask]
    
    if len(feasible_indices) == 0:
        return -1
    
    # Calculate value density for feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = feasible_values / feasible_weights
    densities[~np.isfinite(densities)] = 0.0
    
    # Sort by density descending
    sorted_order = np.argsort(-densities)
    sorted_indices = feasible_indices[sorted_order]
    sorted_weights = feasible_weights[sorted_order]
    sorted_values = feasible_values[sorted_order]
    
    # Lookahead: simulate greedy packing for top candidates
    # Pick the item that leads to the best total value in a quick simulation
    best_item_idx = -1
    best_total_value = -1.0
    
    # Consider top K candidates for lookahead (K is small for speed)
    K = min(20, len(sorted_indices))
    
    for i in range(K):
        candidate_idx = sorted_indices[i]
        candidate_weight = sorted_weights[i]
        candidate_value = sorted_values[i]
        
        # Simulate: take this item first, then greedily add remaining by density
        cap_after = remaining_capacity - candidate_weight
        total_val = candidate_value + _simulate_greedy_completion(cap_after, sorted_weights, sorted_values, skip_index=i)
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_item_idx = candidate_idx
            
        # Time check: if we're getting close to deadline, break early
        if time.time() > _DEADLINE - 5.0:
            break
    
    # If lookahead didn't find anything good, fall back to highest density
    if best_item_idx == -1:
        best_item_idx = sorted_indices[0]
    
    return int(best_item_idx)
# EVOLVE-BLOCK-END
