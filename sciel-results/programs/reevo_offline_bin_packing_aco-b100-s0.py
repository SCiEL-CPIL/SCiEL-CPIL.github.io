import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    Composite heuristic that balances:
    1. Tight fit: minimize remaining gap after placing i+j
    2. Item size: prefer larger items to reduce stranded large items
    3. Bin utilization: reward pairs that fill a high fraction of capacity
    
    Formula: eta[i, j] = (w_j * (capacity - gap)) / (gap + epsilon)
    where gap = capacity - w_i - w_j
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sums of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs
    fits = gaps >= 0
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        # Add epsilon to avoid division by zero for exact fits
        eps = 1e-9
        safe_gaps = np.where(fits, gaps + eps, 1.0)
        
        # Utilization factor: how much of the bin is filled by i+j
        # This rewards tight fits
        utilization = pair_sums / capacity
        
        # Item size factor: prefer placing larger items
        # W[None, :] broadcasts to (n, n) for column j
        item_size = W[None, :]
        
        # Composite: reward tight fits (high utilization) and large items
        # Normalize to avoid overflow
        composite = item_size * utilization / safe_gaps
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, composite, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
