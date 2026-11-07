import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # Sample from the posterior predictive distribution
    # We want to find the candidate with the lowest sample value.
    # The evaluator selects the candidate with the highest acquisition score.
    # Therefore, we return the negative of the sample.
    
    # Generate random normal numbers
    z = np.random.randn(len(mu))
    
    # Compute samples
    samples = mu + sigma * z
    
    # Return negative samples as scores
    return -samples
