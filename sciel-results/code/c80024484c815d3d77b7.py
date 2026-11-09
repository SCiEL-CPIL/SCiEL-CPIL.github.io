# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using raw Expected Improvement (EI) for minimization.
    
    For minimization:
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    - Higher EI means more expected improvement over f_best
    - This is the standard, well-calibrated acquisition function for BO
    
    The score returned is EI (higher is better), which directly guides
    the search toward regions most likely to improve the best observed value.
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu
    
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI calculation
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
