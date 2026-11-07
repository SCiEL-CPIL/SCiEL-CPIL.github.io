"""Rewritten cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: ensure we finish well before 300s
_DEADLINE = time.time() + 280.0  # safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement with an uncertainty bonus.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -20, 20)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: (EI + beta * sigma) / cost
    # The term beta * sigma adds an exploration bonus that is also cost-normalized.
    # This encourages exploring high-uncertainty regions if the cost is low.
    beta = 1.5
    combined = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure all values are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
