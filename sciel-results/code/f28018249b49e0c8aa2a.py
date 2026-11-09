import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # GP-LCB (Lower Confidence Bound) for minimization
    # We want to maximize the score, which corresponds to minimizing the function.
    # Lower bound of f(x) is mu - beta * sigma.
    # We want to find the point with the lowest lower bound.
    # So we maximize - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta controls exploration. 
    # A value of 1.0 is a standard starting point.
    beta = 1.0
    
    # LCB score
    lcb = -mu + beta * sigma
    
    # Cost-aware: divide by cost
    # This favors cheaper evaluations
    scores = lcb / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
