import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # We maximize: -mu + beta * sigma
    # Cost-aware: divide by cost
    
    # Beta controls exploration vs exploitation
    # Higher beta -> more exploration
    beta = 2.0
    
    # Calculate UCB score
    # For minimization, we prefer low mu, so we use -mu
    # We add beta * sigma to encourage exploration in high uncertainty areas
    ucb = -mu + beta * sigma
    
    # Cost-aware scoring
    # cost is shape (2000,)
    scores = ucb / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
