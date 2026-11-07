"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Upper Confidence Bound (UCB).
    
    For minimization, we want to find points with low predicted values.
    UCB for minimization: Lower Bound of prediction = mu - beta * sigma.
    We want to maximize the potential improvement relative to cost.
    Improvement potential ~ f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
        # Fallback: return something simple and fast
        # Prefer low cost and high uncertainty
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB parameter for exploration
    # A value of 1.0 is a standard starting point for UCB
    beta = 1.0
    
    # Calculate UCB score for minimization
    # We want to maximize: (f_best - mu + beta * sigma) / cost
    # This encourages points that are predicted to be low (low mu) 
    # or have high uncertainty (high sigma), divided by cost.
    
    ucb_term = f_best - mu + beta * sigma_safe
    scores = ucb_term / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
