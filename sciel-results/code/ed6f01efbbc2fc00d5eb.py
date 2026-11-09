"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Cost-Adjusted Lower Confidence Bound (LCB).
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure arrays are float64 for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Lower Confidence Bound: score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # We want higher scores to be selected, so we maximize -mu + kappa * sigma
    # But we also want to penalize cost
    
    # Adaptive kappa: balance exploitation and exploration
    # Use a moderate kappa that encourages exploration in uncertain regions
    kappa = 1.5
    
    # LCB score (we want to minimize mu - kappa * sigma, so maximize kappa * sigma - mu)
    lcb_score = kappa * sigma_safe - mu
    
    # Cost penalty: penalize expensive evaluations
    # Normalize cost to [0, 1] for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    if cost_max > cost_min:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost_safe)
    
    # Cost penalty weight
    cost_weight = 0.5
    cost_penalty = cost_weight * cost_norm
    
    # Combined score: LCB score minus cost penalty
    scores = lcb_score - cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
