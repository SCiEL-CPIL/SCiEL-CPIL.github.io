import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Convert to numpy array if not already
    d = np.asarray(demands)
    
    # Calculate pairwise sums
    pair_sums = d[:, None] + d[None, :]
    
    # Initialize heuristic matrix with a very small positive value
    # for pairs that do not fit
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Identify pairs that fit within the capacity
    valid_mask = pair_sums <= capacity
    
    # For valid pairs, calculate the heuristic
    # We want to reward pairs that are both large (high sum) and tight (low waste)
    # Heuristic: Sum / (1 + Waste)
    # Waste = Capacity - Sum
    waste = capacity - pair_sums[valid_mask]
    
    # Calculate the score
    # Avoid division by zero, though waste >= 0 and we add 1, so it's safe.
    scores = pair_sums[valid_mask] / (1.0 + waste)
    
    eta[valid_mask] = scores
    
    return eta
