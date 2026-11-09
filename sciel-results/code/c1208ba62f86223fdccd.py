"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with hypervolume-inspired efficiency metric.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (explore)
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Normalize each component to [0, 1] for fair combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    # Invert since we want lower ucb (better) to have higher score
    ucb_norm = 1.0 - ucb_norm
    
    # Weighted combination
    avg_sigma = np.mean(sigma)
    explore_weight = min(0.4 + avg_sigma * 0.5, 0.7)
    exploit_weight = 1.0 - explore_weight
    
    combined = (ei_norm ** exploit_weight) * (ucb_norm ** explore_weight) * (poi ** 0.2)
    
    # Hypervolume-inspired cost normalization:
    # Use cumulative cost budget as a global efficiency metric
    # This stabilizes exploration-exploitation trade-off over the search trajectory
    total_cost_budget = 30.0  # Given in task: while total cost is below 30
    mean_cost = np.mean(cost)
    
    # Normalize cost relative to the mean, creating a smooth efficiency metric
    # Points with cost below mean get boosted, above mean get penalized
    # This is smoother than raw 1/cost and reflects resource allocation efficiency
    cost_ratio = cost / (mean_cost + 1e-9)
    cost_efficiency = 1.0 / (cost_ratio + 0.5)  # Soft normalization
    
    # Combine with the base acquisition
    scores = combined * cost_efficiency
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're close to timeout, return quickly
    if not _elapsed_ok():
        # Return a simple fallback
        return ei / cost
    
    return scores
# EVOLVE-BLOCK-END
