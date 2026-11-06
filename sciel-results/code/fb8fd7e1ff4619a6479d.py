"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _check_time():
    # Return True if we should continue (time remaining > 1s margin)
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < 119.0

def _greedy_value_fast(capacity, weights, values, excluded_indices):
    """
    Simulates the standard greedy density algorithm on the given arrays,
    excluding the items at excluded_indices.
    Returns the total value accumulated.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Collect non-excluded items
    # Using a list of tuples for sorting
    items = []
    # Optimization: If excluded_indices is small, iterating all n is fine.
    # For N=100, it's trivial.
    for i in range(n):
        if i not in excluded_indices:
            # Pre-filter: if weight > capacity, it can never be picked in this sub-problem
            # But wait, capacity decreases. If it doesn't fit now, it won't fit later.
            # So we can skip it immediately to save sort time?
            # Yes, if w > capacity, it's infeasible for the rest of the greedy fill.
            if weights[i] <= capacity:
                items.append((values[i] / weights[i], values[i], weights[i]))
    
    if not items:
        return 0.0
        
    # Sort by density descending
    # items is list of (density, value, weight)
    items.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    curr_cap = capacity
    
    for density, v, w in items:
        if w <= curr_cap:
            total_val += v
            curr_cap -= w
            if curr_cap <= 0:
                break
        # If w > curr_cap, we skip this item (standard greedy behavior)
        # Note: Standard greedy usually just skips. It does not try to fit smaller items 
        # out of order. The order is strictly by density.
        
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # If time is almost up, fallback to simple greedy
    if not _check_time():
        return int(np.argmax(remaining_values / remaining_weights))
    
    best_total_value = -1.0
    best_index = -1
    
    # 2-Step Lookahead
    # For each candidate i
    for i in range(n):
        if not _check_time():
            break
        w_i = remaining_weights[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = remaining_values[i]
        cap_after_i = remaining_capacity - w_i
        
        # For each candidate j (second item)
        # We can restrict j to those that fit in cap_after_i
        best_val_for_i = -1.0
        best_j = -1
        
        for j in range(n):
            if i == j:
                continue
            w_j = remaining_weights[j]
            if w_j > cap_after_i:
                continue
            
            v_j = remaining_values[j]
            cap_after_j = cap_after_i - w_j
            
            # Simulate greedy fill for the rest
            # Excluded indices: i and j
            # To avoid creating a set/list every time, we can pass a tuple or use a flag
            # Since N is small, creating a set is O(N) which is fine.
            excluded = (i, j)
            
            val_rest = _greedy_value_fast(cap_after_j, remaining_weights, remaining_values, excluded)
            
            total_val = v_i + v_j + val_rest
            
            if total_val > best_val_for_i:
                best_val_for_i = total_val
                best_j = j
        
        # If no valid j was found, best_val_for_i might be -1.0
        # In that case, the best we can do with i is just v_i + greedy_fill(rest)
        # But our loop covers j. If no j fits, we should evaluate i alone + greedy.
        # However, the loop `for j` handles the case where j exists.
        # If no j fits, best_val_for_i remains -1.0.
        # We need to handle the case where only i is picked (and then greedy).
        # Actually, if no j fits, the "2-step" degenerates to "1-step".
        # Let's add a fallback for the "i alone" case if no j was valid, or just always evaluate i alone + greedy as a baseline for i.
        
        # Optimization: The "i alone + greedy" is a subset of the "i + j + greedy" search if we allow j to be "none".
        # But our loop requires j to be a valid item.
        # So let's explicitly calculate the value if we pick i and then do pure greedy.
        val_i_alone = v_i + _greedy_value_fast(cap_after_i, remaining_weights, remaining_values, (i,))
        
        if val_i_alone > best_val_for_i:
            best_val_for_i = val_i_alone
        
        if best_val_for_i > best_total_value:
            best_total_value = best_val_for_i
            best_index = i
            
    if best_index == -1:
        # Fallback to simple greedy if no item fits or something went wrong
        return int(np.argmax(remaining_values / remaining_weights))
        
    return best_index
# EVOLVE-BLOCK-END
