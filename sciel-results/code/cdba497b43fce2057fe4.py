"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian Process posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the Gaussian Process posterior, shape (2000,).
    f_best (float): The lowest observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher is better.
    """
    start_time = time.time()
    
    # Safety margin: ensure we don't exceed any internal limits, though this is O(N)
    # The 300s limit is for the whole loop, but we guard here as requested.
    # If sigma is very small, we add a small epsilon to avoid division issues, 
    # though for UCB we just add beta * sigma.
    
    # UCB: mu + beta * sigma
    # Beta is chosen to balance exploration and exploitation.
    # In high-dimensional spaces (6D), more exploration is needed.
    # Previous experiments suggested beta=2.0 improves over EI.
    # We try beta=2.5 to encourage broader exploration of the rugged landscape.
    beta = 2.5
    
    # Ensure sigma is non-negative (it should be, but just in case)
    sigma = np.maximum(sigma, 1e-8)
    
    scores = mu + beta * sigma
    
    # Check time (unlikely to be an issue for 2000 points)
    if time.time() - start_time > 0.1:
        # Fallback to simple mu if something went wrong, but this shouldn't happen
        pass
        
    return scores
# EVOLVE-BLOCK-END
