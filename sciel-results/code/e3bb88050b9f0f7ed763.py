"""Bounded lookahead 0/1 knapsack heuristic with fractional bound scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.monotonic()
_TOTAL_BUDGET = 115.0

def _remaining_budget():
    return max(0.0, _TOTAL_BUDGET - (time.monotonic() - _START))

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=np.float64)
    v = np.asarray(remaining_values, dtype=np.float64)
    n = len(w)
    if n == 0:
        return -1
    
    # Only consider feasible items
    feasible_mask = w <= cap
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Fast path if near deadline
    if _remaining_budget() < 0.1:
        feasible_v = np.where(feasible_mask, v / np.maximum(w, 1e-12), -np.inf)
        idx = int(np.argmax(feasible_v))
        return idx if feasible_mask[idx] else -1

    # Pre-sort all items by density (v/w) descending
    densities = v / np.maximum(w, 1e-12)
    order = np.argsort(-densities, kind='mergesort')
    
    best_idx = -1
    best_val = -np.inf
    
    # Evaluate each feasible item as the next pick
    # Score = v_i + FractionalKnapsackValue(remaining items, cap - w_i)
    for i in feasible_indices:
        wi = float(w[i])
        vi = float(v[i])
        rem_cap = cap - wi
        
        if rem_cap < 1e-9:
            est_val = vi
        else:
            est_val = vi
            # Simulate fractional knapsack on remaining items
            for j in order:
                if j == i:
                    continue
                wj = float(w[j])
                vj = float(v[j])
                if wj <= rem_cap:
                    est_val += vj
                    rem_cap -= wj
                else:
                    est_val += vj * (rem_cap / wj)
                    break
                
                if rem_cap <= 1e-9:
                    break
        
        if est_val > best_val:
            best_val = est_val
            best_idx = int(i)
            
        if _remaining_budget() < 0.5:
            break

    return best_idx
# EVOLVE-BLOCK-END
