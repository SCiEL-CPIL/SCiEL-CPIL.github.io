import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB-based acquisition function
    # kappa controls the exploration-exploitation trade-off
    kappa = 1.0
    
    # Calculate UCB values (lower is better)
    ucb = mu - kappa * sigma
    
    # Calculate improvement over f_best
    # We want to maximize (f_best - ucb) / cost
    improvement = f_best - ucb
    
    # Ensure cost is positive and avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: higher is better
    scores = improvement / cost_safe
    
    # Return finite scores
    return scores
