"""Cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = evaluate next).
    """
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper points
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-12)
    
    scores = pi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
