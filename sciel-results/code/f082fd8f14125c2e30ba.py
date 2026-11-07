import numpy as np
import time

# Global start time for deadline guard
_PROGRAM_START = time.time()
_DEADLINE = 300.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are close to the limit, switch to pure exploitation
    # to ensure we finish.
    if time.time() - _PROGRAM_START > _DEADLINE - 1.0:
        return -mu
    
    # Ensure sigma is positive to avoid issues with sampling
    # Though sigma from GP should be >= 0
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # Draw a sample from the posterior for each candidate
    # The candidate with the lowest sampled value is the most promising
    # We return -sample so that max(score) selects min(sample)
    samples = mu + sigma * np.random.randn(len(mu))
    scores = -samples
    
    return scores
