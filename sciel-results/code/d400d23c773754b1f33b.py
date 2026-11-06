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
    # Safety margin: return True if we have more than 5 seconds left
    return (time.time() - _START_TIME) < 140.0

def _greedy_fill(cap, weights, values, order):
    """
    Simulate greedy fill following a given order of indices.
    Returns total value achievable.
    """
    total_val = 0.0
    cap_remaining = cap
    for idx in order:
        if cap_remaining <= 0:
            break
        w = weights[idx]
        if w <= cap_remaining:
            cap_remaining -= w
            total_val += values[idx]
    return total_val

def _simulate_after_pick(cap, w_i, v_i, weights, values, density_order, exclude_idx):
    """
    Simulate greedy fill after picking item with index exclude_idx.
    Returns total value including the picked item.
    """
    remaining_cap = cap - w_i
    total_val = v_i
    if remaining_cap <= 0:
        return total_val
    
    cap_rem = remaining_cap
    for j in density_order:
        if j == exclude_idx:
            continue
        w_j = weights[j]
        if w_j <= cap_rem:
            cap_rem -= w_j
            total_val += values[j]
        if cap_rem <= 0:
            break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = remaining_capacity
    
    # Filter out infeasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_weights = weights[feasible_mask]
    feasible_values = values[feasible_mask]
    n_feasible = len(feasible_indices)
    
    # If only one feasible item, take it
    if n_feasible == 1:
        return int(feasible_indices[0])
    
    # Compute density for sorting
    density = feasible_values / feasible_weights
    
    # Sort by density descending
    density_order = np.argsort(-density)
    
    # Also compute other sorting orders for fallback strategies
    order_value = np.argsort(-feasible_values)
    
    # For each candidate first item, simulate the total value
    # if we pick it first, then greedily fill the rest with density order
    best_total_value = -1
    best_first_index = feasible_indices[0]
    
    for i in range(n_feasible):
        if not _check_time():
            break
        
        w_i = feasible_weights[i]
        v_i = feasible_values[i]
        
        total_val = _simulate_after_pick(cap, w_i, v_i, feasible_weights, feasible_values, density_order, i)
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_first_index = feasible_indices[i]
    
    # Also evaluate pure density strategy
    density_total = _greedy_fill(cap, feasible_weights, feasible_values, density_order)
    if density_total > best_total_value:
        best_total_value = density_total
        best_first_index = feasible_indices[density_order[0]]
    
    # Also evaluate pure value strategy
    value_total = _greedy_fill(cap, feasible_weights, feasible_values, order_value)
    if value_total > best_total_value:
        best_total_value = value_total
        best_first_index = feasible_indices[order_value[0]]
    
    return int(best_first_index)
# EVOLVE-BLOCK-END
