import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # demands is a float array
    # capacity is float
    pair = demands[:, None] + demands[None, :]
    gap = capacity - pair
    
    # Initialize with 1e-9
    H = np.full((n, n), 1e-9)
    
    # Mask for valid pairs
    mask = pair <= capacity
    
    # New Heuristic: Large Item, Tight Fit
    # Favors placing a large item j that fits tightly after item i.
    # This mimics Best Fit Decreasing logic where we prioritize large items
    # that minimize the remaining space.
    # H_ij = d_j / (gap + 1)
    
    d_j = demands[None, :] # Shape (1, n)
    
    H[mask] = d_j[mask] / (gap[mask] + 1.0)
    
    # Normalize to [0, 1] for numerical stability
    max_val = H.max()
    if max_val > 0:
        H = H / max_val
        
    return H
