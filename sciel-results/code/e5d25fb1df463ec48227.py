"""Rewritten program using Probability of Improvement (PI) for cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Probability of Improvement (PI).
    
    Args:
        mu: Array of GP means, shape (2000,)
        sigma: Array of GP standard deviations, shape (2000,)
        f_best: Scalar, lowest observed value.
        cost: Array of costs for each candidate, shape (2000,)
        
    Returns:
        Array of scores, shape (2000,), higher is better.
    """
    epsilon = 1e-9
    # Calculate Z-scores: how many standard deviations below f_best the mean is
    z = (f_best - mu) / (sigma + epsilon)
    
    # Probability of Improvement: P(y < f_best)
    pi = norm.cdf(z)
    
    # Cost-aware score: Maximize probability of improvement per unit cost
    # Cost is at least 1, so division is safe and penalizes expensive points
    scores = pi / (cost + epsilon)
    
    # Ensure finite values (though cdf is bounded and cost > 0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
