import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB acquisition for minimization
    # We want to maximize the score.
    # The function to minimize is f.
    # We estimate f ~ N(mu, sigma).
    # We want points with low mu and high sigma.
    # Score = -mu + beta * sigma
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite
    # np.nan_to_num is safe but shouldn't be needed if GP is well-behaved
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
