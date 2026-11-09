import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB for minimization
    # Score = (-mu + beta * sigma) / cost
    # beta controls exploration vs exploitation
    # A value around 1.0 to 3.0 is typical.
    beta = 2.0
    
    # Ensure cost is not zero (it's >= 1, but safe to check)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
