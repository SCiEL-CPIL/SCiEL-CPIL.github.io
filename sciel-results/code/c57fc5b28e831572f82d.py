"""Combined: exact B&B for small instances + 1-step lookahead greedy for large."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.perf_counter()
_DEADLINE = 120.0
_SAFETY = 8.0  # safety margin before 150s kill

def _elapsed():
    return time.perf_counter() - _START

def _bb_optimal_pick(vals, wts, cap, node_cap=50000, time_budget=1.0):
    """
    Branch & bound to find optimal total value and the first item in an optimal solution.
    Returns (opt_value, first_item_local_idx) or (None, None) if budget exceeded.
    """
    n = len(vals)
    if n == 0:
        return 0.0, None
    
    # Sort by density descending
    order = np.argsort(-vals / wts)
    vals = vals[order].astype(float)
    wts = wts[order].astype(float)
    
    # Fractional knapsack upper bound
    def bound(start, cap_left):
        b = 0.0
        i = start
        while i < n and cap_left > 0:
            if wts[i] <= cap_left:
                b += vals[i]
                cap_left -= wts[i]
                i += 1
            else:
                b += vals[i] * (cap_left / wts[i])
                break
        return b
    
    best_val = [0.0]
    best_first = [None]  # local index in sorted array
    count = [0]
    t0 = time.perf_counter()
    
    def rec(start, cap_left, total, first_chosen):
        if count[0] >= node_cap:
            raise RuntimeError("budget")
        count[0] += 1
        if (count[0] & 511) == 0:
            if time.perf_counter() - t0 > time_budget:
                raise RuntimeError("time")
        
        if start >= n:
            if total > best_val[0]:
                best_val[0] = total
                best_first[0] = first_chosen
            return
        
        # Upper bound check
        if total + bound(start, cap_left) <= best_val[0] + 1e-9:
            return
        
        wt = wts[start]
        if wt <= cap_left:
            # Include this item
            fc = start if first_chosen is None else first_chosen
            rec(start + 1, cap_left - wt, total + vals[start], fc)
        
        # Exclude this item
        rec(start + 1, cap_left, total, first_chosen)
    
    try:
        rec(0, float(cap), 0.0, None)
    except RuntimeError:
        return None, None
    
    return best_val[0], best_first[0]

def _simulate_greedy(remaining_capacity, weights, values, sorted_indices, exclude_idx):
    """Simulate greedy fill by pre-sorted density, excluding one item."""
    cap = remaining_capacity
    total_val = 0.0
    for idx in sorted_indices:
        if idx == exclude_idx:
            continue
        w = weights[idx]
        if w <= cap:
            cap -= w
            total_val += values[idx]
            if cap <= 0:
                break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY:
        return -1
    
    remaining_weights = np.asarray(remaining_weights, dtype=float)
    remaining_values = np.asarray(remaining_values, dtype=float)
    n = len(remaining_weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    cap = float(remaining_capacity)
    
    # Find feasible items
    feas_mask = (remaining_weights > 0) & (remaining_weights <= cap)
    feasible_indices = np.where(feas_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    fw = remaining_weights[feasible_indices]
    fv = remaining_values[feasible_indices]
    m = len(feasible_indices)
    
    # For small instances, use exact B&B
    if m <= 26:
        opt_val, first_local = _bb_optimal_pick(fv, fw, cap, node_cap=50000, time_budget=1.5)
        if first_local is not None:
            return int(feasible_indices[first_local])
        # If B&B failed, fall through to heuristic
    
    # For larger instances or B&B failure: 1-step lookahead with greedy completion
    densities = fv / fw
    sorted_local = np.argsort(densities)[::-1]
    sorted_global = feasible_indices[sorted_local]
    
    # Look at top K candidates
    K = min(m, 20)
    
    best_candidate = -1
    best_total = -1.0
    
    for i in range(K):
        cand_global = sorted_global[i]
        w_c = remaining_weights[cand_global]
        val_c = remaining_values[cand_global]
        cap_rem = cap - w_c
        
        # Simulate greedy completion excluding this candidate
        sim_val = _simulate_greedy(cap_rem, remaining_weights, remaining_values,
                                   sorted_global, cand_global)
        total = val_c + sim_val
        
        if total > best_total:
            best_total = total
            best_candidate = cand_global
    
    if best_candidate >= 0:
        return int(best_candidate)
    
    return -1
# EVOLVE-BLOCK-END
