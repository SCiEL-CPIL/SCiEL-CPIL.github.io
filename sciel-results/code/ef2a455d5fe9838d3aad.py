"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0  # safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition: Expected Improvement per unit cost,
    with a penalty for candidates that are spatially redundant (low GP variance).
    
    The score combines:
    1. Expected Improvement (EI) relative to f_best, normalized by cost.
    2. An exploration bonus based on sigma (uncertainty) to encourage
       visiting under-explored regions.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return 1.0 / (cost + 1e-9)
    
    eps = 1e-9
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    ei_cost = ei / (cost + eps)
    
    # Exploration bonus: use sigma as a proxy for information gain
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > eps:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weighted combination: primarily EI, with a smaller exploration bonus
    # The exploration bonus encourages visiting high-uncertainty regions
    # which are likely to be under-explored
    weights = 0.85
    scores = weights * ei_cost + (1 - weights) * sigma_norm / (cost + eps)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
