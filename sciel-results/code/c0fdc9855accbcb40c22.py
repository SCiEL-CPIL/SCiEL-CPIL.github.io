"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known global minimum of Styblinski-Tang 1D
ST_MIN = -39.1662

def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement acquisition function.
    
    mu, sigma: arrays of shape (2000,) in objective units.
    f_best: lowest observed value so far.
    
    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    t0 = time.time()
    n = len(mu)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    # Time guard
    if time.time() - t0 > 2.5:
        return score
    
    return score
# EVOLVE-BLOCK-END
