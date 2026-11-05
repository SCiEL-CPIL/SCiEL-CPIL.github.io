"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _find_lambda(weights, values, capacity, tol=1e-9, max_iter=100):
    """
    Find lambda such that sum of weights of items with value > lambda * weight
    is approximately equal to capacity.
    Uses bisection on lambda.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # If all weights are 0, lambda doesn't matter much, but to be safe return 0
    if max(weights) == 0:
        return 0.0

    # Low bound: 0
    # High bound: max value / min non-zero weight
    min_w = min([w for w in weights if w > 0]) if any(w > 0 for w in weights) else 1.0
    max_v = max(values) if values else 0.0
    
    lo = 0.0
    hi = (max_v / min_w) * 2.0 + 1.0
    
    # If capacity is very large, lambda might be 0
    # If capacity is very small, lambda might be high
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        
        # Count weight of items where value > lambda * weight
        # i.e., value - lambda * weight > 0
        # We want to match capacity.
        # If sum_w > capacity, we need higher lambda (exclude more items)
        # If sum_w < capacity, we need lower lambda (include more items)
        
        # Efficiently compute sum of weights for items with v_i > lambda * w_i
        # Using numpy for speed
        w_arr = np.array(weights)
        v_arr = np.array(values)
        
        # Condition: v_i > lambda * w_i
        mask = v_arr > mid * w_arr
        current_weight = np.sum(w_arr[mask])
        
        if abs(current_weight - capacity) < tol * max(1.0, capacity):
            return mid
            
        if current_weight > capacity:
            lo = mid
        else:
            hi = mid
            
    return (lo + hi) / 2.0

def _greedy_fill_vectorized(weights, values, capacity, exclude_idx):
    """
    Greedily fill the knapsack with remaining items, excluding exclude_idx.
    Returns total value.
    Uses numpy for vectorized operations where possible, but the greedy selection
    is sequential based on density.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
        
    # Create arrays excluding exclude_idx
    if exclude_idx >= 0 and exclude_idx < n:
        indices = np.arange(n)
        mask = indices != exclude_idx
        w = np.array(weights)[mask]
        v = np.array(values)[mask]
    else:
        w = np.array(weights)
        v = np.array(values)
        
    m = len(w)
    if m == 0:
        return 0.0
        
    # Calculate densities
    # Handle zero weights: if w=0, density is inf if v>0.
    # If w=0 and v<=0, density is -inf (or 0, doesn't matter much if we skip)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
        
    # Sort by density descending
    # np.argsort with kind='stable' to be deterministic
    # We want descending order, so we can sort ascending and reverse, or use negative
    # But handling inf is tricky.
    # Let's just sort indices by density descending.
    
    # To handle inf correctly in sorting:
    # np.sort puts inf at the end for ascending, so for descending they come first.
    # But we need to be careful with NaNs if any.
    
    # Let's use a stable sort on negative densities
    # Replace nan with -inf? Or just handle w=0 separately?
    # If w=0, we always take it if v>0.
    
    zero_mask = (w == 0)
    nonzero_mask = ~zero_mask
    
    total_val = 0.0
    remaining_cap = capacity
    
    # Take all zero weight items with positive value
    if np.any(zero_mask):
        zero_vals = v[zero_mask]
        pos_zero_mask = zero_vals > 0
        total_val += np.sum(zero_vals[pos_zero_mask])
        
    # Process non-zero weight items
    if np.any(nonzero_mask):
        w_nz = w[nonzero_mask]
        v_nz = v[nonzero_mask]
        d_nz = v_nz / w_nz
        
        # Sort by density descending
        order = np.argsort(-d_nz, kind='stable')
        
        for idx in order:
            if remaining_cap <= 0:
                break
            wi = w_nz[idx]
            vi = v_nz[idx]
            if wi <= remaining_cap:
                total_val += vi
                remaining_cap -= wi
            else:
                # 0/1 knapsack: if it doesn't fit, we skip it?
                # Standard greedy for 0/1 often just skips if it doesn't fit,
                # but better heuristics might try to fit smaller items later.
                # The "greedy fill" in inspiration 2 likely implies a standard greedy pass.
                # However, a simple greedy that stops when an item doesn't fit is suboptimal.
                # A better "fill" might continue checking if subsequent smaller items fit.
                # Let's assume the standard "take if fits, else skip" for now,
                # or "take if fits, else stop"?
                # Usually "greedy fill" means sort by density and take as many as possible.
                # So we continue iterating.
                pass
                
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack using Lagrangian relaxation to find an item
    that is part of the optimal solution for the current state.
    """
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Global deadline guard
    if not hasattr(select_next_item, '_start_time'):
        select_next_item._start_time = time.time()
    
    elapsed = time.time() - select_next_item._start_time
    if elapsed > 140:  # Safety margin
        return -1

    # Filter feasible items
    feasible_indices = []
    feasible_weights = []
    feasible_values = []
    
    for i in range(n):
        if remaining_weights[i] <= remaining_capacity:
            feasible_indices.append(i)
            feasible_weights.append(remaining_weights[i])
            feasible_values.append(remaining_values[i])
            
    m = len(feasible_indices)
    if m == 0:
        return -1
        
    if m == 1:
        return feasible_indices[0]

    # Find lambda using bisection
    # We want lambda such that the "fractional knapsack" solution using threshold lambda
    # uses up the capacity.
    # Actually, the standard Lagrangian relaxation for knapsack:
    # Max sum v_i x_i
    # s.t. sum w_i x_i <= C
    # x_i in {0,1}
    
    # Dual function g(lambda) = sum max(0, v_i - lambda * w_i) + lambda * C
    # The optimal lambda* minimizes g(lambda).
    # At lambda*, the items with v_i > lambda* w_i are the "core" items.
    # We want to pick an item from this core.
    
    # Let's find lambda* by minimizing g(lambda).
    # g(lambda) is convex and piecewise linear.
    # We can use bisection on the subgradient.
    # Subgradient is sum w_i (for x_i=1) - C.
    # x_i = 1 if v_i > lambda w_i, 0 otherwise.
    # So we want sum w_i (where v_i > lambda w_i) to be close to C.
    
    lambda_star = _find_lambda(feasible_weights, feasible_values, remaining_capacity)
    
    # Identify items in the core: v_i > lambda_star * w_i
    # If there are ties (v_i == lambda_star * w_i), they are partially taken in fractional solution.
    # For 0/1, we should prefer items that are strictly in the core.
    # If all items are in the core, we might need a tie-breaker.
    
    # Calculate "surplus" for each item: v_i - lambda_star * w_i
    # Items with positive surplus are candidates.
    
    candidates = []
    for i in range(m):
        surplus = feasible_values[i] - lambda_star * feasible_weights[i]
        if surplus > 1e-9:
            candidates.append((i, surplus))
            
    if not candidates:
        # If no candidates, pick the one with highest density or value
        # This can happen if lambda is very high or low
        # Fallback to density
        densities = [v / w if w > 0 else float('inf') for w, v in zip(feasible_weights, feasible_values)]
        best_idx = max(range(m), key=lambda i: densities[i])
        return feasible_indices[best_idx]
        
    # We have multiple candidates. Which one to pick?
    # The one that leads to the best total value when we greedily fill the rest?
    # Or the one with the highest surplus?
    # Or the one with the highest density?
    
    # Let's try to pick the item that maximizes the total value of the solution
    # formed by taking that item and then greedily filling the rest with the remaining core items.
    # This is a 1-step look-ahead.
    
    best_item_idx = -1
    best_total_value = -1.0
    
    # To speed this up, we can pre-sort candidates by some heuristic or just iterate.
    # m is at most 100. So iterating is fine.
    
    # For each candidate i:
    # 1. Take item i.
    # 2. Remaining capacity = C - w_i.
    # 3. Remaining items = all other items.
    # 4. Greedily fill remaining capacity with remaining items using density order?
    #    Or using the same lambda?
    #    Using density is a common heuristic.
    
    # Let's use a simple greedy fill for the remaining items.
    # Sort remaining items by density descending.
    
    # Pre-compute densities for all items
    densities = np.zeros(m)
    for i in range(m):
        if feasible_weights[i] > 0:
            densities[i] = feasible_values[i] / feasible_weights[i]
        else:
            densities[i] = float('inf') if feasible_values[i] > 0 else 0.0
            
    # Sort indices by density descending
    # We need to do this for each candidate? No, the relative order of other items doesn't change.
    # But we need to exclude the candidate.
    
    # Let's just iterate through candidates and compute the greedy fill value.
    # To make it faster, we can sort the items once by density, and then for each candidate,
    # we simulate the fill.
    
    # Sort all items by density descending
    sorted_indices = np.argsort(-densities, kind='stable')
    
    for i, _ in candidates:
        rem_cap = remaining_capacity - feasible_weights[i]
        if rem_cap < 0:
            continue
            
        total_val = feasible_values[i]
        
        # Greedy fill with remaining items
        for j in sorted_indices:
            if j == i:
                continue
            if feasible_weights[j] <= rem_cap:
                total_val += feasible_values[j]
                rem_cap -= feasible_weights[j]
            else:
                # Optional: try to find a smaller item that fits?
                # Standard greedy just skips.
                pass
                
        if total_val > best_total_value:
            best_total_value = total_val
            best_item_idx = i
            
    if best_item_idx != -1:
        return feasible_indices[best_item_idx]
        
    # Fallback
    return feasible_indices[candidates[0][0]]

# EVOLVE-BLOCK-END
