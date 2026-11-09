import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior distribution
    # mu and sigma are in objective units
    # We minimize the objective, so we select the candidate with the lowest sampled value.
    # The interface expects higher scores to be preferred, so we return negative samples.
    
    # Handle potential numerical edge cases for sigma
    sigma = np.abs(sigma)
    
    # Generate standard normal random variables
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    
    # Sample from the Gaussian posterior
    samples = mu + sigma * z
    
    # Return negative samples
    return -samples
