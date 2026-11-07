import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB parameter
    # For minimization, we want to maximize -mu + beta * sigma
    # A common choice for beta in 6D problems with small budgets is between 1 and 3.
    # Let's start with 2.0.
    beta = 2.0
    
    # Calculate UCB score
    # Higher score means we should evaluate next
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
