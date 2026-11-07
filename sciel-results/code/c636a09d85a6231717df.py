import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Marginal Thompson Sampling
    # Sample from the marginal posterior at each point
    samples = np.random.normal(mu, sigma)
    
    # We want to minimize f, so lower sample values are better.
    # Return negative samples so higher score = better candidate.
    score = -samples
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Fallback to UCB
        beta = 2.0
        return -mu + beta * sigma
        
    return score
