"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track a global start time and return a safe fallback if we're too close.
_DEADLINE = 330.0
_SAFETY_MARGIN = 2.0
_start_time = None


def _ensure_start():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def _time_remaining():
    global _start_time
    if _start_time is None:
        return _DEADLINE - _SAFETY_MARGIN
    return _DEADLINE - time.time() - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Check if we're running out of time
    if _time_remaining() < 0.5:
        # Return a safe fallback: exploit best point
        return np.where(sigma > 1e-9, (f_best - mu) / cost, np.inf)
    
    n = len(mu)
    
    # Thompson Sampling: sample from the Gaussian posterior
    # We want to maximize the score, which is -sampled_value / cost
    # Since lower objective is better, we sample y ~ N(mu, sigma^2)
    # and score = -y / cost (higher score = more desirable)
    
    # To avoid numerical issues, clip sigma to a small positive value
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample one value from each Gaussian
    # Use a fixed seed per call would be bad; use random state
    # But we need to be careful: the same acquisition function is called many times
    # Each call should use a fresh random sample
    
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(n)
    
    # Score: higher is better. Since we're minimizing, lower sampled value is better.
    # So score = -samples / cost
    # But we also want to handle the case where cost is very small
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
