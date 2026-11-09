import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    # If we are close to the 300s limit, fallback to greedy to ensure completion
    if elapsed > 280:
        return -mu

    # Handle zero or near-zero sigma
    eps = 1e-10
    sigma_safe = np.where(sigma < eps, eps, sigma)
    
    # z-score for improvement
    # We want P(f(x) < f_best) and E[max(0, f_best - f(x))]
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure numerical stability
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
