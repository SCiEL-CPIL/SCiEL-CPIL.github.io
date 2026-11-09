import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Beta controls the trade-off between exploration and exploitation
    # Higher beta -> more exploration
    beta = 3.0
    
    # Calculate UCB score (for minimization, we look for low mu, high sigma)
    # We want to maximize the "potential" which is beta * sigma - mu
    ucb_scores = beta * sigma - mu
    
    # Normalize by cost
    # cost is in [1, 3]
    scores = ucb_scores / cost
    
    # Ensure finite scores
    # If sigma is 0 and mu is inf, etc.
    # But GP should provide finite values.
    # Clip to avoid extreme values if any numerical instability
    # scores = np.clip(scores, -1e6, 1e6)
    
    return scores
