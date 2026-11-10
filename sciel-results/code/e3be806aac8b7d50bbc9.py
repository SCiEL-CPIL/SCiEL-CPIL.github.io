import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety: if we're running out of time, fall back to simple greedy
    elapsed = time.time() - _start_time
    if elapsed > 100:  # 20s safety margin
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])
    
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_values = np.asarray(remaining_values, dtype=np.float64)
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
    
    # If only one feasible item, take it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # For each feasible candidate, simulate greedy completion and compute total value
    best_total = -1
    best_item = -1
    
    n = len(remaining_weights)
    
    for idx in feasible_indices:
        w = remaining_weights[idx]
        v = remaining_values[idx]
        cap = remaining_capacity - w
        
        # Greedy completion with remaining items (excluding idx)
        total_val = v
        # Build remaining arrays without idx
        # For efficiency, work with the feasible set excluding idx
        rem_mask = feasible_mask.copy()
        rem_mask[idx] = False
        rem_w = remaining_weights[rem_mask]
        rem_v = remaining_values[rem_mask]
        
        # Greedy by value density
        if len(rem_w) > 0 and cap > 0:
            # Sort by density descending
            order = np.argsort(rem_v / rem_w)[::-1]
            for i in order:
                if rem_w[i] <= cap:
                    total_val += rem_v[i]
                    cap -= rem_w[i]
                else:
                    # In 0/1 knapsack, we can't take fractional; but for greedy heuristic completion,
                    # we just skip items that don't fit. Actually, we should continue checking
                    # smaller items. The greedy approach: sort by density, take if fits, skip if not.
                    pass
        
        if total_val > best_total:
            best_total = total_val
            best_item = idx
    
    return int(best_item)
