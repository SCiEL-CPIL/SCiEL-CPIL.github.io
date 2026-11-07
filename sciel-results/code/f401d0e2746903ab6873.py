import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Effective Upper Confidence Bound for minimization
    # We want to minimize f(x). 
    # The "optimistic" value is mu - beta * sigma.
    # We want to select the point with the lowest optimistic value per unit cost.
    # Score = -(mu - beta * sigma) / cost
    # Higher score is better.
    
    beta = 3.0
    
    # Ensure sigma is non-negative and handle division by zero in cost
    # cost is in [1, 2], so no zero division risk, but add epsilon for safety
    safe_cost = cost + 1e-9
    
    ucb = mu - beta * sigma
    scores = -ucb / safe_cost
    
    return scores
