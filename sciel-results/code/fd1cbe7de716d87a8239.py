import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # UCB score for minimization: -mu(x) + beta * sigma(x)
    # We maximize this score.
    # To account for cost, we divide by the cost.
    
    # Exploration parameter
    beta = 1.0
    
    # Calculate UCB
    # -mu encourages exploitation (low mean)
    # beta * sigma encourages exploration (high uncertainty)
    ucb = -mu + beta * sigma
    
    # Ensure scores are finite
    # If cost is 0 (shouldn't happen per contract, min cost is 1), handle it
    cost_safe = np.maximum(cost, 1e-8)
    
    score = ucb / cost_safe
    
    return score
