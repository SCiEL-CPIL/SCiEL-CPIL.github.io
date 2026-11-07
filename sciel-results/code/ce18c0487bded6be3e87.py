import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 5s margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    EI estimates the expected reduction in the current best value (f_best)
    if we were to evaluate at a candidate point. It is a form of Knowledge
    Gradient that leverages the current best observation to guide search,
    unlike UCB which ignores f_best.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to a simple exploitation/explore mix if time is tight
        sigma_safe = np.maximum(sigma, 1e-8)
        return -(mu + 0.5 * sigma_safe)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized distance from the best
    # d = (f_best - mu) / sigma
    # If sigma is very small, d is huge.
    # We handle the limit cases implicitly via the math, but to be safe with 
    # numerical stability, we can clip d or rely on scipy's robust handling.
    
    d = (f_best - mu) / sigma_safe
    
    # Calculate CDF (Phi) and PDF (phi)
    # Phi(d) is the probability that a new sample is better than f_best
    # phi(d) is the density at that point
    phi_d = norm.pdf(d)
    Phi_d = norm.cdf(d)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(d) + sigma * phi(d)
    
    ei = (f_best - mu) * Phi_d + sigma_safe * phi_d
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
