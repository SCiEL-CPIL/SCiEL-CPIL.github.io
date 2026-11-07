"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a small adaptive exploration bonus.
    
    Args:
        mu: array of shape (2000,), mean of the posterior
        sigma: array of shape (2000,), standard deviation of the posterior
        f_best: float, lowest observed value so far
    
    Returns:
        Array of shape (2000,) with finite scores; higher is better.
    """
    # Track elapsed time to ensure we don't exceed the deadline
    t_start = time.time()
    
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration bonus: larger when f_best is large (early in optimization),
    # smaller when f_best is close to the minimum (late in optimization)
    # For sphere 1d on [-5, 5], f_best starts around 10-25 and goes to 0.
    # Use a decay function: exploration_weight = 0.1 / (1 + f_best)
    # This gives ~0.05-0.1 early on, and ~0.1 when f_best is near 0.
    exploration_weight = 0.1 / (1.0 + f_best)
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, just return EI
    elapsed = time.time() - t_start
    if elapsed > 295:  # Safety margin before 300s limit
        # Fall back to pure EI which is faster
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
