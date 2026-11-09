import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample from the posterior and return negative sampled values
    # This naturally balances exploration and exploitation
    # For minimization, we want to find points with low values
    # Sample from N(mu, sigma^2) and return negative samples (higher = better for selection)
    
    # Use a fixed seed per call to ensure reproducibility within the evaluation
    # But we want different samples each call, so use time-based seed
    rng = np.random.default_rng()
    
    # Sample from the Gaussian posterior
    # sigma is the standard deviation
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    samples = mu + rng.standard_normal(len(mu)) * sigma_safe
    
    # For minimization, we want to select points with low expected values
    # Return negative samples so that higher scores correspond to lower values
    scores = -samples
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
