import sys
import time
import numpy as np

# Increase recursion limit just in case, though B&B depth is N=100
sys.setrecursionlimit(2000)

_global_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Safety margin: stop if we've used 110s total
    if time.time() - _global_start_time > 110:
        return -1

    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    n = len(weights)
    
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Filter feasible items
    feasible_indices = np.where(weights <= remaining_capacity)[0]
    if len(feasible_indices) == 0:
        return -1
    
    # Prepare items: list of (weight, value, original_index)
    items = []
    for i in feasible_indices:
        items.append((weights[i], values[i], int(i)))
    
    # If only one item, return it
    if len(items) == 1:
        return items[0][2]

    # Sort by density descending for B&B efficiency
    def get_density(item):
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
        
    items.sort(key=get_density, reverse=True)
    
    n_items = len(items)
    
    # Precompute prefix sums for fast fractional bound
    prefix_weights = np.zeros(n_items + 1)
    prefix_values = np.zeros(n_items + 1)
    for i in range(n_items):
        prefix_weights[i+1] = prefix_weights[i] + items[i][0]
        prefix_values[i+1] = prefix_values[i] + items[i][1]
        
    # Helper to calculate fractional knapsack upper bound from index k with capacity cap
    def fractional_bound(k, cap):
        if k >= n_items:
            return 0
        # We want to find j such that prefix_weights[j] - prefix_weights[k] <= cap
        # i.e., prefix_weights[j] <= cap + prefix_weights[k]
        target = cap + prefix_weights[k]
        # np.searchsorted returns the index of the first element > target
        idx_right = np.searchsorted(prefix_weights, target, side='right')
        j = idx_right - 1
        if j < k:
            j = k
        
        # j is the largest index such that prefix_weights[j] <= target
        # This means we can take items k, k+1, ..., j-1 fully.
        val_full = prefix_values[j] - prefix_values[k]
        
        # Remaining capacity
        rem_cap = cap - (prefix_weights[j] - prefix_weights[k])
        
        # If j < n_items, we can take a fraction of item j
        if j < n_items and rem_cap > 0 and items[j][0] > 0:
            val_frac = items[j][1] * (rem_cap / items[j][0])
            return val_full + val_frac
        else:
            return val_full

    # Initial greedy lower bound
    curr_cap = remaining_capacity
    curr_val = 0
    greedy_set = []
    for i in range(n_items):
        w, v, idx = items[i]
        if w <= curr_cap:
            curr_cap -= w
            curr_val += v
            greedy_set.append(idx)
    
    best_val = curr_val
    best_set = greedy_set

    # Time limit for this call
    start_time_call = time.time()
    time_limit_call = 1.0
    
    # Recursive B&B with in-place list modification
    def solve(k, cap, val, chosen):
        nonlocal best_val, best_set
        
        if time.time() - start_time_call > time_limit_call:
            return False # Timeout
            
        # Prune
        if val + fractional_bound(k, cap) <= best_val:
            return True
            
        if k == n_items:
            if val > best_val:
                best_val = val
                best_set = chosen[:] # Copy
            return True

        w, v, idx = items[k]

        # Branch 1: Include item k
        if w <= cap:
            chosen.append(idx)
            if not solve(k + 1, cap - w, val + v, chosen):
                chosen.pop()
                return False
            chosen.pop()
            
        # Branch 2: Exclude item k
        if not solve(k + 1, cap, val, chosen):
            return False
            
        return True

    solve(0, remaining_capacity, 0, [])

    if best_set:
        # Return the index of the first item in the optimal set
        # Using min for determinism
        return min(best_set)
    
    return -1
