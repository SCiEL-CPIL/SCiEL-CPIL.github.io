"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return safe scores
    if time.time() - _start_time > _TIMEOUT:
        return np.ones_like(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle any remaining numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-aware acquisition: EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
