import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score, so we return -LCB.
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score(x) = -mu(x) + beta * sigma(x)
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # sigma is non-negative, mu is finite.
    # If sigma is very small, it's fine.
    
    return scores
