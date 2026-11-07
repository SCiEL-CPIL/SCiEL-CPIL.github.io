"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) for Branin-2D minimization.
    
    EI balances exploitation (low mu) and exploration (high sigma) in a principled way.
    For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    Parameters:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP std devs
        f_best: lowest observed function value
    
    Returns:
        (2000,) array of scores, higher = more desirable
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for minimization
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores, clip negatives to zero
    score = np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)
    
    return score
# EVOLVE-BLOCK-END
