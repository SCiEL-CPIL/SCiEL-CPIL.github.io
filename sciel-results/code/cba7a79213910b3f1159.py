import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    # The score is the negative of the sampled value (higher = lower sampled value = better)
    # Use a single random draw for consistency across all candidates
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within a call
    eps = rng.standard_normal(mu.shape[0])
    
    # Sample from the posterior: mu + sigma * eps
    samples = mu + sigma * eps
    
    # For minimization, we want the lowest sampled value
    # So the score should be -sample (higher score = lower sampled value)
    scores = -samples
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
