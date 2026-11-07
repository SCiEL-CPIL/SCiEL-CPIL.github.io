"""Cost-aware acquisition for Powell 4D BO with non-uniform evaluation costs."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    Uses Probability of Improvement (PI) with a multiplicative cost penalty.
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:
        # Fallback: simple EI / cost
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Apply cost penalty via exponentiation
    # Score = PI^(1/cost)
    # Since cost >= 1, 1/cost <= 1.
    # This reduces the score for expensive points more than cheap points,
    # especially when PI < 1.
    # If PI is close to 1, the effect is minimal.
    # If PI is close to 0, the effect is strong (score stays near 0).
    
    # Add a small epsilon to cost to avoid division by zero (though cost >= 1)
    cost_safe = np.maximum(cost, 1e-6)
    
    score = np.power(pi, 1.0 / cost_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
