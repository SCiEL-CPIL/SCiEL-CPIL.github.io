"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition with a high exploration constant.
    
    Parameters:
        mu: array of shape (2000,) with posterior mean values
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: scalar, the best (lowest) observed function value so far
    
    Returns:
        scores: array of shape (2000,) with acquisition scores (higher = better)
    """
    # Safety margin for computation time
    start_time = time.time()
    
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB: we want to minimize mu - k * sigma, so for "higher is better" scoring
    # we return -mu + k * sigma (negating the LCB objective)
    # Using a high exploration constant to aggressively explore uncertain regions
    k = 3.0
    
    scores = -mu + k * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
