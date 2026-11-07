import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing problem.
    
    Args:
        demands: array of item sizes (floats).
        capacity: bin capacity (float).
        
    Returns:
        matrix: n x n matrix where entry (i, j) is the attractiveness of 
                placing item j after item i.
    """
    n = len(demands)
    demands_arr = np.array(demands, dtype=np.float64)
    
    # Sort items to compute future compatibility counts
    sorted_demands = np.sort(demands_arr)
    
    # For each item j, compute how many other items can fit in the residual space
    # after placing j in a bin (residual = capacity - demands[j])
    # We want to reward transitions (i, j) where j leaves room for many future items
    
    # For each j, count how many items k (k != j) satisfy demands[j] + demands[k] <= capacity
    # This is equivalent to: demands[k] <= capacity - demands[j]
    # Using binary search on sorted_demands
    
    future_counts = np.zeros(n, dtype=np.float64)
    for j in range(n):
        residual = capacity - demands_arr[j]
        # Count how many items can fit in residual space
        # We exclude item j itself from the count if it fits (but actually we're counting
        # other items that could be placed after j, so we count all items that fit,
        # and the actual exclusion of j happens implicitly in the ACO selection)
        # Actually, we want to count items k where k != j and demands[k] <= residual
        # Since demands[j] might be <= residual, we need to be careful
        # Let's count all items that fit, then subtract 1 if j itself fits
        idx = np.searchsorted(sorted_demands, residual, side='right')
        count = idx
        # If item j itself fits in the residual, subtract 1 to exclude it
        if demands_arr[j] <= residual:
            count -= 1
        future_counts[j] = count
    
    # Now compute the heuristic matrix
    # H[i, j] should be high when:
    # 1. items i and j fit together (demands[i] + demands[j] <= capacity)
    # 2. item j leaves room for many future items (high future_counts[j])
    # 3. the pair uses space efficiently (low waste)
    
    pair_sums = demands_arr[:, None] + demands_arr[None, :]
    fit_mask = pair_sums <= capacity
    waste = capacity - pair_sums
    
    # Base heuristic: reward tight fits and high future compatibility
    # Combine: 1/(1+waste) * (1 + future_counts[j])
    # This way, j with high future_counts gets boosted
    
    H = np.full((n, n), 1e-9)
    
    if np.any(fit_mask):
        # For fitting pairs, compute heuristic
        waste_fit = waste[fit_mask]
        # Get the column indices (j) for the fitting pairs
        rows, cols = np.where(fit_mask)
        future_boost = 1.0 + future_counts[cols]
        
        # Heuristic: 1/(1+waste) * future_boost
        H[fit_mask] = (1.0 / (1.0 + waste_fit)) * future_boost
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
