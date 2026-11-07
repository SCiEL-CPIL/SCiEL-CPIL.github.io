import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Lower Confidence Bound for minimization)
    # We want to minimize the function.
    # The optimistic lower bound is mu - beta * sigma.
    # We want to pick the point with the lowest lower bound.
    # So we return a score that is high for low lower bounds.
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
