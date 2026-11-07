import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0 # Safety margin from 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (KG proxy)
    scaled by 1/sqrt(cost).
    """
    # Deadline check: if we are running out of time, return a safe fallback
    # (e.g., uniform scores or zero scores) to avoid timeout.
    # Since this function is called frequently, we check elapsed time.
    if time.time() - _START_TIME > _DEADLINE:
        # Return zeros, which will likely lead to picking the first candidate or random if tie-breaking is unstable.
        # Better to return the previous best? We don't have state.
        # Returning zeros is safe and fast.
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (which is equivalent to KG for single-step improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost adjustment: sublinear discount 1/sqrt(cost)
    # Cost is always >= 1.0
    cost_adj = 1.0 / np.sqrt(cost)
    
    scores = ei * cost_adj
    
    # Clean up any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
