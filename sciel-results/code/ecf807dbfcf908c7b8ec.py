import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Cost-Weighted Upper Confidence Bound (UCB) for minimization
    # We want to maximize: (beta * sigma - mu) / cost
    # This balances exploration (high sigma) and exploitation (low mu)
    # beta controls the exploration-exploitation trade-off
    beta = 1.5
    
    # Ensure cost is positive to avoid division errors
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate the UCB score
    # For minimization, we prefer low mu and high sigma.
    # The term (beta * sigma - mu) is high when mu is low or sigma is high.
    ucb_value = beta * sigma - mu
    scores = ucb_value / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
