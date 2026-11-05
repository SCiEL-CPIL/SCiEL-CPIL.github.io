"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin before 120s/150s limit


def _greedy_fill_value(capacity, weights, values, exclude_idx=-1):
    """
    Estimate the value of filling the knapsack with remaining items
    using a greedy density strategy, optionally excluding one item.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    n = len(weights)
    
    # Create mask for all items except exclude_idx
    if exclude_idx >= 0 and exclude_idx < n:
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
    else:
        mask = np.ones(n, dtype=bool)
    
    if not np.any(mask):
        return 0.0
    
    w_rem = weights[mask]
    v_rem = values[mask]
    
    # Calculate densities
    safe_w = np.where(w_rem > 1e-12, w_rem, 1e-12)
    densities = v_rem / safe_w
    
    # Sort by density descending
    order = np.argsort(densities)[::-1]
    
    total_val = 0.0
    rem_cap = capacity
    
    for idx in order:
        w = w_rem[idx]
        v = v_rem[idx]
        if w <= rem_cap:
            total_val += v
            rem_cap -= w
            if rem_cap <= 1e-12:
                break
    
    return total_val


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add, or -1 to stop.
    
    Strategy: 
    1. Check time deadline first.
    2. Filter feasible items.
    3. For small problem sizes, use a "lookahead" heuristic:
       For each candidate item, estimate the total value if we pick it now
       and then fill the rest greedily by density. Pick the candidate
       that yields the highest estimated total value.
    4. For larger problem sizes, fall back to a refined density-based score
       with utilization adjustment to keep runtime low.
    """
    # 1. Time guard
    if time.time() > _GLOBAL_DEADLINE:
        return -1
    
    # 2. Basic setup
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    cap = float(remaining_capacity)
    
    n = len(weights)
    if n == 0 or cap <= 1e-12:
        return -1
    
    # Filter feasible items
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_idx = np.where(feasible_mask)[0]
    f_weights = weights[feasible_idx]
    f_values = values[feasible_idx]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # 3. Handle zero-weight items immediately (always take highest value)
    zero_mask = f_weights <= 1e-12
    if np.any(zero_mask):
        zero_local = np.where(zero_mask)[0]
        best_zero_local = zero_local[np.argmax(f_values[zero_local])]
        return int(feasible_idx[best_zero_local])
    
    # 4. Decide between lookahead and simple heuristic based on size
    # Lookahead is O(n^2) roughly. For n < 100, it's fast enough.
    # The task says 100-item instances, so n will start at 100 and decrease.
    # 100^2 = 10,000 operations per call. With ~100 calls, that's 1M ops.
    # This is very fast in Python/NumPy.
    
    if f_n <= 150:
        # Lookahead strategy
        # For each feasible item i, calculate:
        # value_if_pick_i = values[i] + greedy_fill(cap - weights[i], all_items, exclude=i)
        
        best_total = -1.0
        best_local_idx = 0
        
        # Pre-compute densities for the greedy fill part to speed up?
        # The _greedy_fill_value function sorts each time. 
        # We can optimize by sorting once and passing the sorted order, 
        # but for n=100, sorting is cheap.
        
        for i in range(f_n):
            w_i = f_weights[i]
            v_i = f_values[i]
            rem_cap = cap - w_i
            
            # If rem_cap is very small, no other items fit
            if rem_cap <= 1e-12:
                est_total = v_i
            else:
                # We need to exclude item i from the greedy fill.
                # Note: f_weights/f_values are the feasible subset.
                # The greedy fill should consider ALL remaining items that fit in rem_cap.
                # However, items in the original 'weights' array that were not feasible 
                # for 'cap' might be feasible for 'rem_cap'? No, rem_cap < cap, 
                # so if an item didn't fit in cap, it doesn't fit in rem_cap.
                # So we only need to consider the feasible set f_weights/f_values.
                
                # We pass the local index i to exclude.
                est_total = v_i + _greedy_fill_value(rem_cap, f_weights, f_values, exclude_idx=i)
            
            if est_total > best_total:
                best_total = est_total
                best_local_idx = i
        
        return int(feasible_idx[best_local_idx])
        
    else:
        # Fallback for large n (though unlikely given 100-item instances)
        # Use refined density score
        safe_weights = np.where(f_weights > 1e-12, f_weights, 1e-12)
        density = f_values / safe_weights
        utilization = f_weights / cap
        score = density * (1.0 + 0.1 * utilization)
        best_local_idx = int(np.argmax(score))
        return int(feasible_idx[best_local_idx])

# EVOLVE-BLOCK-END
