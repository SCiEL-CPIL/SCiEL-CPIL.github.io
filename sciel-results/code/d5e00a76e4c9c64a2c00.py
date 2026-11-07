"""Rewritten program for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track wall-clock deadline
_START_TIME = time.time()
_DEADLINE_MARGIN = 2.0  # Seconds to leave before hard stop

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Normalized Expected Improvement (z-EI).
    
    This approach computes the Expected Improvement (EI) and normalizes it by
    the posterior standard deviation (sigma). This results in a score that
    depends only on the z-score of the candidate's mean relative to the best
    observed value.
    
    Score = (f_best - mu) * Phi(z) / sigma + phi(z)
          = z * Phi(z) + phi(z)
    where z = (f_best - mu) / sigma.
    
    This metric represents the expected improvement in units of standard
    deviations. It is robust to the scale of the objective function and
    provides a smooth, monotonic ranking of candidates based on their
    potential to improve upon f_best relative to their uncertainty.
    
    Returns scores where higher values indicate better candidates.
    """
    # Check time to ensure we don't exceed limits
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    # If we are extremely close to the deadline, use a very fast heuristic
    # Fallback: simple z-score ranking
    if remaining_time < _DEADLINE_MARGIN:
        eps = 1e-8
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        # For large positive z, z*Phi(z) ~ z. For large negative z, ~ 0.
        # We just return z as a proxy, which is fast.
        scores = np.where(np.isfinite(z), z, -np.inf)
        return scores

    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # norm.cdf and norm.pdf are vectorized and handle large/small z well
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Calculate Normalized EI: z * Phi(z) + phi(z)
    # This is equivalent to EI / sigma
    scores = z * Phi_z + phi_z
    
    # Ensure all scores are finite
    # z * Phi(z) can be problematic if z is inf, but we clamped sigma.
    # If mu is inf, z is inf.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
