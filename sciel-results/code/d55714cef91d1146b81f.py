import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior distribution
    # Higher score means more likely to be the minimum
    
    # Clip sigma to avoid division by zero or negative variance issues, 
    # though GP sigma should be positive.
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw samples from the Gaussian posterior
    # shape: (2000,)
    samples = mu + sigma * np.random.randn(len(mu))
    
    # We want to minimize f. 
    # So we want to select the sample with the lowest value.
    # The function returns scores where higher is better.
    # Therefore, we return the negative of the samples.
    return -samples
