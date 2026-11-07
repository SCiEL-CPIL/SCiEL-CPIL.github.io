"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    
    # Safety margin: if we're getting close to the 300s limit, return EI to be fast
    if elapsed > 280:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use a hybrid acquisition: blend of Expected Improvement and a 
    # distance-weighted exploration term to encourage exploration in 
    # early iterations while exploiting in later ones.
    
    # Compute standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small amount of pure uncertainty to encourage exploration
    # This helps escape local minima in the Michalewicz function
    # Weight the exploration term to be small but non-zero
    exploration_weight = 0.1
    score = ei + exploration_weight * sigma
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
