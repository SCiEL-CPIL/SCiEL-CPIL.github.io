"""Rewritten acquisition function using Cost-Weighted Upper Confidence Bound (UCB) for minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    For minimization, we want to find points with low predicted values and high uncertainty.
    We maximize the score: (-mu + beta * sigma) / cost.
    This is equivalent to minimizing the Lower Confidence Bound (LCB) of the function value,
    normalized by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far (unused in pure UCB, but kept for interface)
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Exploration parameter beta
    # Since targets are standardized in the GP, mu and sigma are on similar scales.
    # A value of 1.0 is a standard starting point.
    beta = 1.0
    
    # Calculate UCB score for minimization
    # We want to maximize this score.
    # Lower predicted value (low mu) is good -> -mu is high.
    # Higher uncertainty (high sigma) is good -> beta * sigma is high.
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware scoring: Score per unit cost
    eps = 1e-8
    cost_safe = np.maximum(cost, eps)
    
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
