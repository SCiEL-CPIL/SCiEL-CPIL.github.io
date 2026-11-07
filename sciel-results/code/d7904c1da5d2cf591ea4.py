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
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Fallback: return -mu to ensure finite output and exploit best known
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robustness: ensure sigma is strictly positive to avoid division by zero
    # If sigma is 0, the model is certain. 
    # If mu < f_best, z = inf, cdf = 1.
    # If mu > f_best, z = -inf, cdf = 0.
    # This is correct behavior for PoI.
    # However, numerical stability: use a small epsilon.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # If mu is much smaller than f_best, z is large positive, cdf ~ 1.
    # If mu is much larger than f_best, z is large negative, cdf ~ 0.
    # If sigma is large, z is closer to 0, cdf ~ 0.5 (uncertain).
    
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
