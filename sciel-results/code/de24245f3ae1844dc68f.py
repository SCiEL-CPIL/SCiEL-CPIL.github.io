import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling:
    # Draw a sample from the posterior predictive distribution: N(mu, sigma^2)
    # sample = mu + sigma * z, where z ~ N(0, 1)
    # We want to minimize, so we prefer lower samples.
    # The score should be higher for better candidates.
    # So score = -sample = -(mu + sigma * z)
    
    # Generate random standard normal variables
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Compute the sample
    sample = mu + sigma * z
    
    # Return negative sample as score (higher score = lower predicted value = better)
    return -sample
