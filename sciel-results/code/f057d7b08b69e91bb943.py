"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # Safety margin before 150s

def _vectorized_greedy_fill(cap, sorted_weights, sorted_values, exclude_pos):
    """
    Greedy fill over sorted items (by density desc), skipping item at exclude_pos.
    Uses cumulative sums for speed. Returns total value of greedy completion.
    """
    n = len(sorted_weights)
    if n <= 1:
        return 0.0
    
    cap = float(cap)
    if cap <= 1e-12:
        return 0.0
    
    # Prefix sums of weights and values
    prefix_w = np.zeros(n + 1)
    prefix_v = np.zeros(n + 1)
    np.cumsum(sorted_weights, out=prefix_w[1:])
    np.cumsum(sorted_values, out=prefix_v[1:])
    
    total_val = 0.0
    remaining_cap = cap
    
    # Process items before exclude_pos
    if exclude_pos > 0:
        # Find how many items from the start fit
        # Items 0..exclude_pos-1 have total weight prefix_w[exclude_pos]
        # Find largest k <= exclude_pos such that prefix_w[k] <= remaining_cap
        k_before = np.searchsorted(prefix_w, remaining_cap, side='right') - 1
        k_before = min(k_before, exclude_pos)
        if k_before > 0:
            total_val += prefix_v[k_before]
            remaining_cap -= prefix_w[k_before]
    
    # Process items after exclude_pos
    if exclude_pos < n - 1:
        # Items exclude_pos+1 .. n-1
        # Their weights in order: sorted_weights[exclude_pos+1:]
        # Cumulative weight from position exclude_pos+1:
        # weight of items exclude_pos+1..j is prefix_w[j+1] - prefix_w[exclude_pos+1]
        # We want largest j such that prefix_w[j+1] - prefix_w[exclude_pos+1] <= remaining_cap
        # i.e., prefix_w[j+1] <= remaining_cap + prefix_w[exclude_pos+1]
        threshold = remaining_cap + prefix_w[exclude_pos + 1]
        # Find largest j+1 such that prefix_w[j+1] <= threshold, with j+1 >= exclude_pos+2
        # j+1 ranges from exclude_pos+2 to n
        # searchsorted on prefix_w for threshold
        pos = np.searchsorted(prefix_w, threshold, side='right') - 1
        pos = max(pos, exclude_pos + 1)  # at least up to exclude_pos+1 (zero items after)
        pos = min(pos, n)
        if pos > exclude_pos + 1:
            total_val += prefix_v[pos] - prefix_v[exclude_pos + 1]
    
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        feasible = np.where(remaining_weights <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = remaining_values[feasible] / remaining_weights[feasible]
        return int(feasible[np.argmax(densities)])

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # 1. Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # 2. Sort feasible items by value/weight density (descending)
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    densities = feasible_values / feasible_weights
    
    # Sort by density descending, break ties by value descending
    # Use lexsort: last key is primary
    # We want descending density, then descending value
    # lexsort sorts ascending, so negate for descending
    order = np.lexsort((-feasible_values, -densities))
    
    sorted_global_indices = feasible_indices[order]
    sorted_weights = remaining_weights[sorted_global_indices]
    sorted_values = remaining_values[sorted_global_indices]
    
    # Precompute prefix sums once
    prefix_w = np.zeros(len(sorted_weights) + 1)
    prefix_v = np.zeros(len(sorted_weights) + 1)
    np.cumsum(sorted_weights, out=prefix_w[1:])
    np.cumsum(sorted_values, out=prefix_v[1:])
    
    # 3. Determine K (number of candidates to look ahead)
    # With vectorized fill, we can afford much larger K
    K = min(len(sorted_global_indices), 200)
    
    best_candidate_idx = -1
    best_total_value = -1.0
    
    # 4. For each of the top K candidates, simulate the rest using prefix sums
    for i in range(K):
        candidate_idx = sorted_global_indices[i]
        
        val_c = sorted_values[i]
        w_c = sorted_weights[i]
        
        cap_rem = remaining_capacity - w_c
        if cap_rem < 0:
            continue
        
        # Simulate greedy completion: skip item at position i in sorted order
        sim_val = _vectorized_greedy_fill_from_prefix(cap_rem, sorted_weights, sorted_values, prefix_w, prefix_v, i)
        
        total_val = val_c + sim_val
        
        if total_val > best_total_value:
            best_total_value = total_val
            best_candidate_idx = candidate_idx

    if best_candidate_idx == -1:
        # Fallback: pick the highest density feasible item
        return int(feasible_indices[np.argmax(densities)])

    return int(best_candidate_idx)

def _vectorized_greedy_fill_from_prefix(cap, sorted_weights, sorted_values, prefix_w, prefix_v, exclude_pos):
    """
    Greedy fill over sorted items (by density desc), skipping item at exclude_pos.
    Uses precomputed prefix sums. Returns total value of greedy completion.
    """
    n = len(sorted_weights)
    if n <= 1:
        return 0.0
    
    cap = float(cap)
    if cap <= 1e-12:
        return 0.0
    
    total_val = 0.0
    remaining_cap = cap
    
    # Process items before exclude_pos: positions 0..exclude_pos-1
    if exclude_pos > 0:
        # Find largest k <= exclude_pos such that prefix_w[k] <= remaining_cap
        k_before = np.searchsorted(prefix_w, remaining_cap, side='right') - 1
        k_before = min(k_before, exclude_pos)
        if k_before > 0:
            total_val += prefix_v[k_before]
            remaining_cap -= prefix_w[k_before]
    
    # Process items after exclude_pos: positions exclude_pos+1..n-1
    if exclude_pos < n - 1:
        # Weight of items from exclude_pos+1 to j is prefix_w[j+1] - prefix_w[exclude_pos+1]
        # We want largest j such that prefix_w[j+1] - prefix_w[exclude_pos+1] <= remaining_cap
        # i.e., prefix_w[j+1] <= remaining_cap + prefix_w[exclude_pos+1]
        threshold = remaining_cap + prefix_w[exclude_pos + 1]
        # j+1 ranges from exclude_pos+2 to n
        pos = np.searchsorted(prefix_w, threshold, side='right') - 1
        pos = max(pos, exclude_pos + 1)
        pos = min(pos, n)
        if pos > exclude_pos + 1:
            total_val += prefix_v[pos] - prefix_v[exclude_pos + 1]
    
    return total_val
# EVOLVE-BLOCK-END
