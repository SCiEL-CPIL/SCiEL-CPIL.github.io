import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid sampling errors
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling:
    # Draw a sample from the posterior distribution N(mu, sigma^2) for each candidate.
    # Since the goal is to minimize the function value, we want to select the candidate
    # with the lowest sampled value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the samples.
    samples = np.random.normal(mu, sigma)
    
    return -samples
