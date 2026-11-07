"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Draws a single random sample from the posterior distribution at each candidate point:
    f_s = mu + sigma * xi, where xi ~ N(0,1)
    
    Returns -f_s as the score (higher score = lower sampled value = better).
    This stochastic approach naturally balances exploration and exploitation.
    """
    # Time guard: if we are close to the deadline, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single random sample from the posterior at each candidate point
    # xi ~ N(0, 1) for each of the 2000 candidates
    xi = np.random.standard_normal(mu.shape)
    
    # Sampled function value
    f_s = mu + sigma_safe * xi
    
    # Return negative sampled value as score (higher is better for minimization)
    score = -f_s
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
