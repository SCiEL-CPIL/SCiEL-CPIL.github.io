"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 330s wall-clock; use 280s safety margin
_DEADLINE = time.time() + 280.0


def _safe_finite(arr, fallback=0.0):
    """Replace non-finite values with fallback."""
    return np.where(np.isfinite(arr), arr, fallback)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    Sample y ~ N(mu, sigma) for each candidate and return (f_best - y) / cost.
    This stochastically balances exploration and exploitation without
    requiring complex normalization or heuristic beta parameters.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    n = len(mu)
    
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample one function value per candidate
    # y ~ N(mu, sigma)
    y_samples = mu + sigma_safe * np.random.randn(n)
    
    # Score: improvement over f_best divided by cost
    # Higher means better (more improvement per unit cost)
    scores = (f_best - y_samples) / cost_safe
    
    # Ensure finite
    scores = _safe_finite(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
