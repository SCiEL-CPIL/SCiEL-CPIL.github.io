import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # Sample from the posterior distribution N(mu, sigma^2)
    # Select the candidate with the lowest sampled value.
    # Since the loop picks the highest score, we return -sampled_value.
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw samples
    samples = np.random.normal(mu, sigma)
    
    # Invert for minimization (higher score = lower value)
    scores = -samples
    
    # Handle any potential NaNs/Infs from numerical issues
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
