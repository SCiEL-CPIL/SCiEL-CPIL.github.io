"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _fractional_bound(weights, values, capacity, start_idx, sorted_indices):
    """
    Calculate the fractional knapsack upper bound starting from a specific index
    in a pre-sorted list of items (by density).
    Optimized to break early.
    """
    if capacity <= 0 or start_idx >= len(sorted_indices):
        return 0.0
    
    total_val = 0.0
    remaining_cap = capacity
    
    # Iterate through items in density order
    for i in range(start_idx, len(sorted_indices)):
        idx = sorted_indices[i]
        w = weights[idx]
        v = values[idx]
        
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            # Fractional part
            total_val += (remaining_cap / w) * v
            break
            
    return total_val


def _branch_and_bound(weights, values, capacity):
    """
    Solve the 0/1 knapsack problem exactly using Branch and Bound.
    Returns the maximum value achievable.
    Optimized with precomputed suffix sums for faster pruning.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Sort items by density (value/weight) descending
    # Handle zero weights? Problem statement implies positive weights usually, but let's be safe.
    # If weight is 0 and value > 0, it should always be taken.
    # Assuming standard knapsack constraints: w > 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.divide(values, weights, out=np.zeros_like(values, dtype=float), where=weights>0)
    
    sorted_indices = np.argsort(-densities)
    
    # Pre-sort weights and values into local arrays for faster access in the loop
    sorted_w = weights[sorted_indices].astype(np.float64)
    sorted_v = values[sorted_indices].astype(np.float64)
    
    # Precompute suffix sums of weights and values for quick upper bound checks
    # suffix_w[i] = sum(sorted_w[i:])
    # suffix_v[i] = sum(sorted_v[i:])
    suffix_w = np.zeros(n + 1)
    suffix_v = np.zeros(n + 1)
    if n > 0:
        suffix_w[n-1] = sorted_w[n-1]
        suffix_v[n-1] = sorted_v[n-1]
        for i in range(n-2, -1, -1):
            suffix_w[i] = suffix_w[i+1] + sorted_w[i]
            suffix_v[i] = suffix_v[i+1] + sorted_v[i]
    
    best_value = 0.0
    
    # Stack elements: (next_item_index, current_weight, current_value)
    # We start by considering the item at index 0 in the sorted list.
    stack = [(0, 0.0, 0.0)]
    
    while stack:
        i, curr_w, curr_v = stack.pop()
        
        # Prune if current weight exceeds capacity (shouldn't happen if we check before push)
        if curr_w > capacity:
            continue
            
        # Update best value
        if curr_v > best_value:
            best_value = curr_v
            
        # Check if we've processed all items
        if i >= n:
            continue
            
        # Calculate upper bound for this node
        # If the sum of all remaining weights fits in the remaining capacity, 
        # we can take all of them. This is the tightest possible bound and saves iteration.
        rem_cap = capacity - curr_w
        if suffix_w[i] <= rem_cap:
            ub = curr_v + suffix_v[i]
            if ub <= best_value:
                continue
            # If we can take all remaining, we can update best immediately?
            # No, because we might have found a better combination earlier? 
            # Actually, if we take all remaining, that's a valid solution.
            # But we still need to explore? No, if we take all, that's the max for this branch.
            # However, B&B explores include/exclude. If we take all, we don't need to branch further.
            # So we can just update best and not push children.
            if ub > best_value:
                best_value = ub
            continue
        
        # If even the first item doesn't fit? 
        # The fractional bound handles this naturally (returns 0 if capacity < w[i] and we don't take fraction? 
        # Wait, fractional bound takes a fraction. 
        # If sorted_w[i] > rem_cap, the bound is curr_v + (rem_cap/sorted_w[i])*sorted_v[i].
        
        # Standard fractional bound calculation
        # We can optimize this by checking if the bound is tight enough.
        ub = curr_v + _fractional_bound(sorted_w, sorted_v, rem_cap, i, np.arange(n))
        
        if ub <= best_value:
            continue
            
        # Branch 1: Exclude item i
        # Push exclusion branch first so it's processed later (LIFO)
        stack.append((i + 1, curr_w, curr_v))
        
        # Branch 2: Include item i
        new_w = curr_w + sorted_w[i]
        if new_w <= capacity:
            new_v = curr_v + sorted_v[i]
            stack.append((i + 1, new_w, new_v))
            
    return best_value


def _greedy_fill(cap, weights, values, sorted_indices, exclude_set):
    """
    Fast greedy fill for fallback or quick estimation.
    """
    total_val = 0.0
    rem_cap = cap
    for idx in sorted_indices:
        if idx in exclude_set:
            continue
        w = weights[idx]
        if w <= rem_cap:
            rem_cap -= w
            total_val += values[idx]
            if rem_cap <= 1e-12:
                break
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item using Branch and Bound lookahead.
    Returns index into remaining arrays, or -1 to stop.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # If very few items, just solve directly?
    # For N=100, B&B is fast.
    
    # Sort feasible items by density
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.divide(feasible_values, feasible_weights, out=np.zeros_like(feasible_values, dtype=float), where=feasible_weights>0)
        
    sorted_local = np.argsort(-densities)
    
    # We consider the top K candidates for the first pick.
    # Dynamic K: start small, increase if time allows? 
    # Given the strict time limit and 8 instances, we need to be conservative.
    # 100 items. B&B on 99 items is very fast (<1ms usually).
    # 10 candidates * 1ms = 10ms per call.
    # 100 calls * 10ms = 1s. Very safe.
    # Let's try K=15 to be a bit more thorough.
    K = min(15, len(feasible_indices))
    top_k_local = sorted_local[:K]
    
    best_total_value = -1
    best_global_idx = -1
    
    # Pre-calculate some info
    for local_idx in top_k_local:
        # Check time
        if time.time() - _START_TIME > _TIME_LIMIT - 2.0:
            # Fallback to simple density pick
            # Pick the highest density item among the rest
            # Actually, just return the best found so far or the first one
            if best_global_idx != -1:
                return int(best_global_idx)
            return int(feasible_indices[0])

        global_idx = feasible_indices[local_idx]
        w_pick = remaining_weights[global_idx]
        v_pick = remaining_values[global_idx]
        
        rem_cap = remaining_capacity - w_pick
        if rem_cap < 0:
            continue
            
        # Prepare arrays for the sub-problem (excluding the picked item)
        # We create a mask of items that are feasible and not the picked one
        sub_mask = feasible_mask.copy()
        sub_mask[global_idx] = False
        
        # Get indices of sub-problem items
        sub_indices = np.where(sub_mask)[0]
        
        if len(sub_indices) == 0:
            total_val = v_pick
        else:
            sub_weights = remaining_weights[sub_indices]
            sub_values = remaining_values[sub_indices]
            
            # Solve the sub-problem using B&B
            # If time is running out, we might want to use a faster heuristic
            if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
                # Fallback to greedy for speed
                sub_val = _greedy_fill(rem_cap, sub_weights, sub_values, 
                                       np.argsort(-sub_values/sub_weights), set())
                total_val = v_pick + sub_val
            else:
                sub_val = _branch_and_bound(sub_weights, sub_values, rem_cap)
                total_val = v_pick + sub_val
                
        if total_val > best_total_value:
            best_total_value = total_val
            best_global_idx = global_idx
            
    return int(best_global_idx)

# EVOLVE-BLOCK-END
