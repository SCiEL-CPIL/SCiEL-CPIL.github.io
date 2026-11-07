import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Upper Confidence Bound (UCB) divided by cost^gamma.
    This encourages exploration (via sigma) and exploitation (via -mu)
    while penalizing high-cost evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # UCB parameter. 
    # A value around 2.0-3.0 is typical for exploration.
    # Since we are minimizing, we want high sigma and low mu.
    # Score to maximize: beta * sigma - mu
    beta = 2.5
    
    # Cost penalty exponent. 
    # Previous success with 1.5. Let's stick with it or try 1.2.
    # Let's try 1.5 first as it was measured to be better than 1.0.
    gamma = 1.5
    
    # Calculate UCB score
    ucb_score = beta * sigma - mu
    
    # Apply cost penalty
    cost_penalty = np.power(cost, gamma)
    
    score = ucb_score / cost_penalty
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
