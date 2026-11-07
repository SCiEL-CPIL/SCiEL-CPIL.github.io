import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound).
    
    Args:
        mu: Array of shape (2000,) with GP mean predictions.
        sigma: Array of shape (2000,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with acquisition scores.
    """
    # Deadline guard: if this function takes too long, return a simple fallback
    # The limit is 300s, but we aim for < 1s normally.
    # We don't have a global timer, but we can check if the computation is stuck?
    # Actually, vectorized ops are fast. We'll just ensure no infinite loops.
    
    # Avoid division by zero in cost
    cost_safe = np.where(cost < 1e-6, 1e-6, cost)
    
    # Avoid division by zero in z calculation if used, but UCB doesn't use z directly
    # UCB Score = beta * sigma - mu
    # We want to minimize the function, so we want low mu.
    # Score should be high for low mu and high sigma.
    # Score = -mu + beta * sigma
    
    beta = 2.0
    
    scores = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
