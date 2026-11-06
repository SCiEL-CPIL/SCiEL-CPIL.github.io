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
    # Safety margin: return True if we have more than 15 seconds left
    return (time.time() - _START_TIME) < 130.0

def _greedy_fill_value(weights, values, order, cap):
    """
    Simulate greedy fill following a given order of indices.
    Returns total value achievable.
    """
    if cap <= 0:
        return 0.0
    total_val = 0.0
    cap_rem = cap
    for idx in order:
        if cap_rem <= 0:
            break
        w = weights[idx]
        if w <= cap_rem:
            cap_rem -= w
            total_val += values[idx]
    return total_val

def _fast_greedy_fill(weights, values, density_order, cap, skip_idx=None):
    """
    Fast greedy fill using precomputed cumulative arrays.
    Skip one item if needed.
    """
    if cap <= 0:
        return 0.0
    
    n = len(weights)
    total_val = 0.0
    cap_rem = cap
    
    for pos in range(n):
        if cap_rem <= 0:
            break
        idx = density_order[pos]
        if idx == skip_idx:
            continue
        w = weights[idx]
        if w <= cap_rem:
            cap_rem -= w
            total_val += values[idx]
    
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
    density_order = np.argsort(-density, kind='mergesort')
    
    # Precompute: for each position in density order, store the global feasible index
    # density_order[pos] gives the index in feasible arrays at position pos in density order
    
    best_total_value = -1.0
    best_first_index = feasible_indices[0]
    
    # 1-step lookahead: for each candidate first item, simulate greedy fill on rest
    # Use early termination: if remaining capacity after picking item i is very small,
    # the greedy fill won't add much, so we can bound the search
    
    # Precompute prefix sums of weights and values in density order for potential speedup
    # But since we skip one item, we'll use a different approach:
    # For each candidate i, the greedy fill on remaining items is equivalent to
    # greedy fill on all items minus item i. We can compute this efficiently.
    
    # Strategy: evaluate all candidates but with early exit if value can't beat current best
    for i in range(n_feasible):
        if not _check_time():
            break
        
        w_i = feasible_weights[i]
        v_i = feasible_values[i]
        remaining_cap = cap - w_i
        
        if remaining_cap <= 0:
            total_val = v_i
        else:
            total_val = v_i
            cap_rem = remaining_cap
            # Iterate through density-sorted items, skipping item i
            # Use local variables for speed
            fw = feasible_weights
            fv = feasible_values
            do = density_order
            nf = n_feasible
            for pos in range(nf):
                if cap_rem <= 0:
                    break
                orig_idx = do[pos]
                if orig_idx == i:
                    continue
                w_j = fw[orig_idx]
                if w_j <= cap_rem:
                    cap_rem -= w_j
                    total_val += fv[orig_idx]
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_first_index = feasible_indices[i]
    
    # Also evaluate alternative strategies
    # Strategy 1: Pure density
    if n_feasible > 0:
        density_first_idx = density_order[0]
        density_first_global = feasible_indices[density_first_idx]
        density_total = _greedy_fill_value(feasible_weights, feasible_values, density_order, cap)
        if density_total > best_total_value:
            best_total_value = density_total
            best_first_index = density_first_global
    
    # Strategy 2: Pure value
    if n_feasible > 0:
        value_order = np.argsort(-feasible_values, kind='mergesort')
        value_first_idx = value_order[0]
        value_first_global = feasible_indices[value_first_idx]
        value_total = _greedy_fill_value(feasible_weights, feasible_values, value_order, cap)
        if value_total > best_total_value:
            best_total_value = value_total
            best_first_index = value_first_global
    
    # Strategy 3: Weight ascending
    if n_feasible > 0:
        weight_order = np.argsort(feasible_weights, kind='mergesort')
        weight_first_idx = weight_order[0]
        weight_first_global = feasible_indices[weight_first_idx]
        weight_total = _greedy_fill_value(feasible_weights, feasible_values, weight_order, cap)
        if weight_total > best_total_value:
            best_total_value = weight_total
            best_first_index = weight_first_global
    
    # 2-step lookahead for top 3 candidates from 1-step
    # Pick top 3 by 1-step value, then for each, try each second item and greedy fill
    if _check_time() and n_feasible > 2:
        # Re-evaluate top candidates with 2-step lookahead
        # First, get the top 3 candidates from 1-step evaluation
        # We need to re-compute or store the 1-step values
        # For efficiency, just re-evaluate a few promising candidates
        
        # Get candidates: top by density, top by value, and the current best
        candidate_set = set()
        candidate_set.add(density_order[0])
        if len(density_order) > 1:
            candidate_set.add(density_order[1])
        candidate_set.add(np.argmax(feasible_values))
        candidate_set.add(np.argmin(feasible_weights))
        candidate_set.add(best_first_index if best_first_index in feasible_indices else density_order[0])
        
        for first_i in candidate_set:
            if not _check_time():
                break
            if first_i >= n_feasible:
                continue
            
            w_i = feasible_weights[first_i]
            v_i = feasible_values[first_i]
            remaining_cap = cap - w_i
            if remaining_cap <= 0:
                continue
            
            # For each possible second item, simulate greedy fill
            for second_j in range(n_feasible):
                if second_j == first_i:
                    continue
                w_j = feasible_weights[second_j]
                if w_j > remaining_cap:
                    continue
                v_j = feasible_values[second_j]
                cap_after_two = remaining_cap - w_j
                
                if cap_after_two <= 0:
                    total_val = v_i + v_j
                else:
                    total_val = v_i + v_j
                    cap_rem = cap_after_two
                    fw = feasible_weights
                    fv = feasible_values
                    do = density_order
                    nf = n_feasible
                    for pos in range(nf):
                        if cap_rem <= 0:
                            break
                        orig_idx = do[pos]
                        if orig_idx == first_i or orig_idx == second_j:
                            continue
                        w_k = fw[orig_idx]
                        if w_k <= cap_rem:
                            cap_rem -= w_k
                            total_val += fv[orig_idx]
                
                if total_val > best_total_value:
                    best_total_value = total_val
                    best_first_index = feasible_indices[first_i]
    
    return int(best_first_index)
# EVOLVE-BLOCK-END
