# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    EI is a principled acquisition function that balances exploration and exploitation
    by maximizing the expected improvement over the current best value.
    
    For minimization:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    This function returns the raw EI values, which are higher for points with
    higher expected improvement.
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu

    # Handle near-zero sigma to avoid division issues
    # Use a small epsilon to prevent division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (for minimization)
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    # In case of numerical issues, replace non-finite values with 0
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
