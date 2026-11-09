import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # score = -mu + beta * sigma
    # We want to minimize the function, so we maximize -mu + beta * sigma
    # beta controls the exploration/exploitation trade-off
    
    # Adaptive beta: 
    # As f_best gets closer to 0 (the minimum), we want to exploit more (lower beta).
    # As f_best is high, we want to explore more (higher beta).
    # However, sigma naturally decreases in explored regions.
    # A fixed beta is often sufficient and robust.
    # Let's try a beta that is slightly high to ensure we don't get stuck in local minima 
    # of the Ackley function which has many local minima.
    
    beta = 2.5
    
    # Handle zero sigma to avoid division issues if any (though UCB doesn't divide)
    # But for numerical stability in other metrics, good practice.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    return scores
