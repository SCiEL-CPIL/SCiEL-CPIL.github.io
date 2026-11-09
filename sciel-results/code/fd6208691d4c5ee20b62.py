import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior distribution N(mu, sigma^2)
    # Since we minimize, we want the lowest sample.
    # Return negative samples so that max(score) corresponds to min(sample).
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    z = np.random.randn(mu.shape[0])
    samples = mu + sigma * z
    
    # Return negative to invert the optimization direction
    return -samples
