import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 290.0 # Leave 10s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: UCB with beta=1
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 1.0 * sigma_safe

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate alpha
    # Avoid division by zero, though sigma_safe handles it
    alpha = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(alpha) + sigma * phi(alpha)
    # Where Phi is CDF and phi is PDF of standard normal
    cdf = norm.cdf(alpha)
    pdf = norm.pdf(alpha)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Return EI as score (higher is better)
    return ei
