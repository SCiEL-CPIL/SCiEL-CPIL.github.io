# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a global start time and a safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 115s safety margin within 120s limit


def _lookahead_score(remaining_capacity, remaining_weights, remaining_values):
    """
    For each feasible item, simulate picking it, then greedily fill the rest
    by value density. Return the total value for each candidate.
    """
    n = len(remaining_weights)
    if n == 0:
        return np.array([])
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter to feasible items (weight <= remaining_capacity)
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return np.array([])
    
    feasible_indices = np.where(feasible_mask)[0]
    fw = w[feasible_mask]
    fv = v[feasible_mask]
    fn = len(fw)
    
    if fn == 0:
        return np.array([])
    
    # Compute densities for all feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = fv / fw
        # Handle zero-weight items: if w==0 and v>0, density is inf
        # We'll handle them specially
    
    # For each candidate i, simulate:
    # 1. Pick item i
    # 2. Remaining capacity: c - fw[i]
    # 3. Greedily fill remaining capacity with other feasible items sorted by density
    
    # Sort feasible items by density (descending)
    # Items with inf density (w=0, v>0) go first
    order = np.argsort(-densities, kind='mergesort')
    
    # Precompute prefix sums for the density-sorted order
    sorted_w = fw[order]
    sorted_v = fv[order]
    
    # Prefix sums of weights and values in density order
    prefix_w = np.cumsum(sorted_w)
    prefix_v = np.cumsum(sorted_v)
    
    # For each candidate i, we need to find how many of the other items
    # (excluding i) we can fit in the remaining capacity
    # This is tricky because we need to exclude item i from the sorted list
    
    # Alternative approach: for each candidate, do a quick greedy fill
    # Since n is at most 100, this is O(n^2) which is fine
    
    scores = np.zeros(fn)
    
    for idx in range(fn):
        # Candidate is feasible_indices[order[idx]]? No, let's restructure.
        # Actually, let's iterate over original feasible indices.
        pass
    
    # Let me restructure: iterate over each feasible item as candidate
    scores = np.zeros(fn)
    
    for i in range(fn):
        # Pick item i
        rem_cap = c - fw[i]
        if rem_cap < 0:
            scores[i] = fv[i]
            continue
        
        total_val = fv[i]
        
        # Now greedily fill with remaining items (all feasible except i)
        # Sort the other items by density
        other_indices = np.concatenate([np.arange(i), np.arange(i+1, fn)])
        other_densities = densities[other_indices]
        other_order = np.argsort(-other_densities, kind='mergesort')
        
        for j in other_order:
            orig_idx = other_indices[j]
            if fw[orig_idx] <= rem_cap:
                rem_cap -= fw[orig_idx]
                total_val += fv[orig_idx]
            else:
                # Could potentially take a fractional part, but this is 0/1 knapsack
                # In lookahead, we just skip items that don't fit
                # But actually for a better heuristic, we might want to consider
                # that skipping a heavy item might allow lighter items
                # However, the density greedy fill already handles this by trying
                # in order. We just skip if it doesn't fit.
                pass
        
        scores[i] = total_val
    
    return scores


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Check deadline
    if time.time() > _DEADLINE:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter to feasible items
    feasible_mask = w <= c
    if not np.any(feasible_mask):
        return -1
    
    # Use lookahead scoring
    try:
        scores = _lookahead_score(c, w, v)
        if len(scores) == 0:
            return -1
        
        # scores is indexed by position in the feasible subset
        feasible_indices = np.where(feasible_mask)[0]
        best_local = int(np.argmax(scores))
        return int(feasible_indices[best_local])
    except Exception:
        # Fallback to density greedy
        feasible_indices = np.where(feasible_mask)[0]
        fw = w[feasible_indices]
        fv = v[feasible_indices]
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = fv / fw
        best_local = int(np.argmax(densities))
        return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
