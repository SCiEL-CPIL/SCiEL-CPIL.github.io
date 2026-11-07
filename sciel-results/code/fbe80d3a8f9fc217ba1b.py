import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    Uses Thompson Sampling: draws a sample from the GP posterior at each candidate
    and selects the candidate with the lowest sampled value.
    Since the interface requires higher scores to be better, we return the negative
    of the sampled value.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw standard normal noise for each candidate
    # This implements the sampling from N(mu, sigma^2)
    z = np.random.randn(mu.shape[0])
    
    # Sampled values: y = mu + sigma * z
    # We are minimizing f(x). We want to pick x with lowest y.
    # The interface wants us to return scores where HIGHER is better.
    # So we return -y.
    scores = -mu - sigma * z
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
