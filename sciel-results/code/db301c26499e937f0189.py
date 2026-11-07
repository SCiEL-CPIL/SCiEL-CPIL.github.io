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
    
    # Initialize with a very small value for infeasible transitions
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute pairwise sums
    pair_sums = demands_arr[:, None] + demands_arr[None, :]
    
    # Feasibility mask: items i and j must fit together
    # Note: In the actual ACO loop, the ant checks if item j fits in the *current bin*.
    # The current bin's remaining space is not just capacity - demands[i].
    # However, the heuristic is static. We use the pairwise fit as a proxy.
    # If demands[i] + demands[j] > capacity, they can never be in the same bin 
    # if i was the first item. If there were other items, it's even less likely.
    # So this is a safe feasibility constraint for the heuristic.
    fit_mask = pair_sums <= capacity
    
    # Get indices of feasible pairs
    rows, cols = np.where(fit_mask)
    
    if len(rows) > 0:
        # Calculate waste for feasible pairs
        # waste = capacity - demands[i] - demands[j]
        waste = capacity - pair_sums[rows, cols]
        
        # Heuristic components:
        # 1. Tightness: 1 / (1 + waste). Rewards small waste.
        # 2. Item Size: demands[j]. Rewards larger items (harder to place later).
        
        # Combine them
        # H[i, j] = demands[j] / (1 + waste)
        
        values = demands_arr[cols] / (1.0 + waste)
        
        # Ensure values are positive and finite
        values[~np.isfinite(values)] = 1e-9
        values[values <= 0] = 1e-9
        
        H[rows, cols] = values

    # Final safety check
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
