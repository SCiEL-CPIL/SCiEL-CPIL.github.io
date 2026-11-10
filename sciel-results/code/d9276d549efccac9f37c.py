import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # demands is a float array
    # capacity is float
    pair = demands[:, None] + demands[None, :]
    gap = capacity - pair
    
    # Initialize with 1e-9
    H = np.full((n, n), 1e-9)
    
    # Mask for valid pairs (fitting items)
    mask = pair <= capacity
    
    # New Heuristic: Inverse Gap (Tight Fit Reward)
    # H = 1 / (gap + 1)
    # This approach favors pairs that result in the smallest remaining space,
    # encouraging bins to be filled as tightly as possible.
    # This is distinct from size-product heuristics as it ignores the absolute
    # magnitude of the items and focuses purely on the efficiency of the fit.
    
    H[mask] = 1.0 / (gap[mask] + 1.0)
    
    # Normalize to [0, 1] for numerical stability and standardization
    # This does not change relative probabilities in ACO
    max_val = H.max()
    if max_val > 0:
        H = H / max_val
        
    return H
