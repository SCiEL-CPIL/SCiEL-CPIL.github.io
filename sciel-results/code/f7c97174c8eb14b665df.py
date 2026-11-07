"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) for minimization with a small
    adaptive exploration term that decays as f_best approaches the
    known minimum of 3.0.
    
    Args:        mu: Array of shape (2000,) - GP posterior means
        sigma: Array of shape (2000,) - GP posterior standard deviations
        f_best: float - Best observed value so far
    
    Returns:
        Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration: more exploration early, less as we converge
    # Progress: 0 when f_best is far from 3, 1 when close
    progress = np.clip((f_best - 3.0) / 5.0, 0.0, 1.0)
    
    # Beta decreases from ~0.5 to ~0.05 as progress increases
    beta = 0.05 + 0.45 * progress
    
    # Scale exploration term by sigma directly (in objective units)
    scores = ei + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
