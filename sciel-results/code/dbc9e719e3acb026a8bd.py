"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 150s. We want to stay well under.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin


def _greedy_fill(capacity, weights, values, exclude_idx=None):
    """
    Greedily fill capacity with items sorted by density (value/weight).
    Returns the total value achieved.
    exclude_idx: if provided, skip this index (in the local array).
    """
    if capacity <= 0:
        return 0.0
    n = len(weights)
    if n == 0:
        return 0.0
    
    # Create indices to sort by density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, 0)
    
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    for idx in order:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        if weights[idx] <= remaining_cap:
            total_value += values[idx]
            remaining_cap -= weights[idx]
        if remaining_cap <= 0:
            break
    
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item index (into the remaining arrays) to add, or -1 to stop.
    
    Strategy: 1-step lookahead with greedy future estimation.
    For each feasible item, estimate the total value if we pick it now and then
    greedily fill the remaining capacity. Pick the item with the highest total.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter to feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # For each feasible item, compute a lookahead score
    best_score = -1
    best_idx = -1
    
    for local_i in range(len(feasible_indices)):
        # Skip if deadline approaching
        if time.time() > _DEADLINE - 0.5:
            break
        
        w_i = feasible_weights[local_i]
        v_i = feasible_values[local_i]
        cap_after = remaining_capacity - w_i
        
        if cap_after < 0:
            continue
        
        # Compute greedy future value from remaining items (excluding local_i)
        future_value = _greedy_fill(cap_after, feasible_weights, feasible_values, exclude_idx=local_i)
        
        total_score = v_i + future_value
        
        if total_score > best_score:
            best_score = total_score
            best_idx = int(feasible_indices[local_i])
    
    if best_idx == -1:
        # Fallback: pick highest density feasible item
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(feasible_weights > 0, feasible_values / feasible_weights, 0)
        best_density = -1
        best_idx = -1
        for local_i in range(len(feasible_indices)):
            if densities[local_i] > best_density:
                best_density = densities[local_i]
                best_idx = int(feasible_indices[local_i])
    
    return best_idx
# EVOLVE-BLOCK-END
