"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Convert to numpy array for vectorized operations
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sum of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate remaining capacity (gap) for each pair
    # If pair_sum > capacity, gap is negative (invalid)
    gaps = capacity - pair_sums
    
    # Initialize heuristic matrix
    # We want high values for small positive gaps (tight fit)
    # For invalid pairs (gap <= 0), we use a very small value
    # For valid pairs, we use 1 / (gap + epsilon)
    # This creates a strong preference for items that fill the bin completely
    
    epsilon = 1e-9
    
    # Mask for valid pairs (gap >= 0)
    fit_mask = gaps >= 0
    
    # Compute 1 / (gap + epsilon) for valid pairs
    safe_gaps = np.where(fit_mask, gaps + epsilon, 1.0)
    attractiveness = 1.0 / safe_gaps
    
    # Ensure invalid pairs are 1e-9
    H = np.where(fit_mask, attractiveness, 1e-9)
    
    # Check deadline (unlikely to be hit, but required by directive)
    if time.time() > deadline:
        # Fallback to simple size-based heuristic if something went wrong
        return np.where(fit_mask, W[None, :], 1e-9)
        
    return H
# EVOLVE-BLOCK-END
