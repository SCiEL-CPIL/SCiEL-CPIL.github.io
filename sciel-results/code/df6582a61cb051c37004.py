"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0 - 30.0  # 30s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: scalar, lowest observed value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), finite scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a safe fallback
    if _time_left() < 0.1:
        return np.zeros_like(mu)

    # Guard against edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Probability of improvement beyond a threshold epsilon
    # P(f(x) < f_best - epsilon) = Phi((f_best - epsilon - mu) / sigma)
    # Use a small epsilon to focus on meaningful improvements
    epsilon = 0.01
    
    z = (f_best - epsilon - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    prob_improve = norm.cdf(z_clipped)
    
    # Ensure probability is non-negative
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # Cost-normalized acquisition: probability per unit cost
    scores = prob_improve / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
