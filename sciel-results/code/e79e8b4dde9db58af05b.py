"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin, using a complementarity-score approach.
    
    The heuristic is based on how well two items complement each other in terms
    of filling a bin. Items that together use most of the capacity (but don't
    exceed it) are considered highly complementary. We use a sorted-order
    two-pointer sweep to identify near-optimal complementary pairs and assign
    high attractiveness to those, while penalizing large deviations.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Sort demands to identify complementarity structure
    sorted_demands = np.sort(demands)
    
    # For each item, find its ideal partner using two-pointer approach
    # An ideal partner for item with demand d is one with demand close to capacity - d
    # such that d + partner <= capacity
    
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid pairs are those where sum <= capacity
    valid = pair_sums <= capacity
    
    # For valid pairs, compute how close they are to filling the bin
    # Higher complementarity = closer to capacity without exceeding
    # Normalize: (pair_sum / capacity) gives fill ratio for valid pairs
    fill_ratio = np.where(valid, pair_sums / capacity, 0.0)
    
    # For invalid pairs, set fill_ratio to 0
    # For valid pairs, we want high attractiveness for high fill ratio
    
    # Now, use the sorted structure to identify "ideal" complementary pairs
    # For each item in sorted order, the ideal partner is the largest item
    # that fits with it (two-pointer approach)
    
    # Build the ideal partner mapping using two pointers on sorted array
    ideal_partner = np.zeros(n, dtype=np.int64)
    ideal_complement_value = np.zeros(n, dtype=np.float64)
    
    # For each unique demand value, find its best complement in sorted order
    # We'll work with sorted indices
    sorted_indices = np.argsort(demands)
    sorted_vals = demands[sorted_indices]
    
    # Two-pointer: for each i (small end), find j (large end) such that
    # sorted_vals[i] + sorted_vals[j] <= capacity and is maximized
    # For each i, the best j is the largest index such that the sum fits
    
    # Precompute for each position in sorted array, the best complement
    best_complement_idx = np.zeros(n, dtype=np.int64)
    best_complement_val = np.zeros(n, dtype=np.float64)
    
    j = n - 1
    for i in range(n):
        # Move j down until sorted_vals[i] + sorted_vals[j] <= capacity
        while j >= 0 and sorted_vals[i] + sorted_vals[j] > capacity:
            j -= 1
        if j >= 0:
            best_complement_idx[i] = j
            best_complement_val[i] = sorted_vals[i] + sorted_vals[j]
        else:
            best_complement_idx[i] = -1
            best_complement_val[i] = 0.0
    
    # Now, for each original item, determine its ideal complement value
    # Map back from sorted position to original index
    # For each original index, find its position in sorted order
    sort_pos = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        sort_pos[sorted_indices[idx]] = idx
    
    # For each original item i, its ideal complement value is best_complement_val[sort_pos[i]]
    ideal_comp_val = best_complement_val[sort_pos]
    
    # The ideal partner demand for item i is ideal_comp_val[i] - demands[i]
    ideal_partner_demand = ideal_comp_val - demands[None, :]  # shape (n, n)
    
    # For each pair (i, j), compute how close demands[j] is to the ideal partner demand for i
    # Only for valid pairs
    deviation = np.abs(demands[None, :] - ideal_partner_demand)
    
    # For valid pairs, compute a score based on:
    # 1. Fill ratio (higher is better)
    # 2. Closeness to ideal complement (lower deviation is better)
    
    # Scale deviation: max possible deviation is capacity
    dev_norm = deviation / capacity
    
    # Combine: high fill ratio and low deviation
    # Use exponential decay on deviation
    dev_score = np.exp(-3.0 * dev_norm)
    
    # Final heuristic: for valid pairs, combine fill ratio and deviation score
    # For invalid pairs, set to 1e-9
    heuristic = np.where(valid, fill_ratio * dev_score, 1e-9)
    
    # Ensure diagonal (item with itself) is handled - if 2*demands[i] <= capacity, it's valid
    # This is already handled by the valid mask
    
    # Ensure all values are finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
