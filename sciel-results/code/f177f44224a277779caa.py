import numpy as np
import time

def _pareto_dp_complete(capacity, weights, values):
    """
    Exact DP using Pareto frontier pruning for float weights.
    Returns max value achievable with given capacity.
    """
    if len(values) == 0 or capacity <= 0:
        return 0.0
    
    # State: list of (weight, value) tuples, non-dominated
    states = [(0.0, 0.0)]
    
    for i in range(len(weights)):
        w_item = weights[i]
        v_item = values[i]
        
        # Generate new states by adding this item
        new_states = []
        for sw, sv in states:
            new_w = sw + w_item
            if new_w <= capacity:
                new_states.append((new_w, sv + v_item))
        
        # Combine old and new states
        all_states = states + new_states
        
        # Prune dominated states
        # Sort by weight ascending, then by value descending
        all_states.sort(key=lambda x: (x[0], -x[1]))
        
        pruned = []
        max_val_so_far = -1.0
        last_weight = None
        
        for w, v in all_states:
            # If this state has a value greater than any state with lower or equal weight, keep it
            # Note: Since sorted by weight asc, we just need to check if v > max_val_so_far
            # But if weights are equal, we keep the one with highest value.
            # The sort handles this: for same weight, highest value comes first.
            if v > max_val_so_far:
                pruned.append((w, v))
                max_val_so_far = v
        
        states = pruned
        
        # Optional: Early exit if only one state or states list is very small?
        # No, we need to process all items.
        
    return max(v for _, v in states) if states else 0.0

def _greedy_complete(capacity, weights, values):
    """Greedy complete using value density, return total value."""
    if len(values) == 0 or capacity <= 0:
        return 0.0
    total = 0.0
    cap = capacity
    w = weights.copy()
    v = values.copy()
    n = len(w)
    for _ in range(n):
        if cap <= 0:
            break
        if len(w) == 0:
            break
        # Pick best value density among feasible
        feasible = w <= cap
        if not np.any(feasible):
            break
        idxs = np.where(feasible)[0]
        densities = v[idxs] / w[idxs]
        best = idxs[np.argmax(densities)]
        total += v[best]
        cap -= w[best]
        w = np.delete(w, best)
        v = np.delete(v, best)
    return total

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
    
    best_total = -1.0
    best_idx = -1
    
    # Time guard
    start_time = time.time()
    timeout = 100.0 # Keep 20s margin for 120s limit
    
    # Heuristic: If there are many items, DP might be slow.
    # But capacity is small (12.5), so Pareto frontier should be small.
    # Let's try DP for all feasible items.
    
    for idx in feasible_idx:
        if time.time() - start_time > timeout:
            # Fallback to greedy if we're running out of time
            # Just pick the best greedy estimate for the remaining feasible items
            # Or simply break and return best found so far
            break
            
        # Pick this item
        new_cap = remaining_capacity - remaining_weights[idx]
        new_val = remaining_values[idx]
        
        # Remove this item
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        new_w = remaining_weights[mask]
        new_v = remaining_values[mask]
        
        # Complete using Exact Pareto DP
        if new_cap > 0 and len(new_w) > 0:
            # Check if we should use DP or Greedy based on item count?
            # With 100 items, DP is O(N*K). If K is small, it's fast.
            # Let's use DP.
            total = new_val + _pareto_dp_complete(new_cap, new_w, new_v)
        else:
            total = new_val
        
        if total > best_total:
            best_total = total
            best_idx = idx

    if best_idx == -1:
        # Fallback if timeout happened before any evaluation or all failed
        # Pick best density
        densities = remaining_values[feasible_idx] / remaining_weights[feasible_idx]
        best_idx = feasible_idx[np.argmax(densities)]

    return int(best_idx)
