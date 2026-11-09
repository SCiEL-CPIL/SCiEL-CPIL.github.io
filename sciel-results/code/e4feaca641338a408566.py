"""Rewritten acquisition function using Cost-Weighted Probability of Improvement (PI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI) normalized by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive to avoid division by zero in z-score
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Probability of Improvement
    # We want P(f(x) < f_best)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Ensure PI is within [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware scoring: PI per unit cost
    cost_safe = np.maximum(cost, eps)
    
    scores = pi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
