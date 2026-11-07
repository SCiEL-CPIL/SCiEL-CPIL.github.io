import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Thompson Sampling to balance exploration and exploitation.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure sigma is non-negative for sampling
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Generate random samples from the posterior distribution
    # z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    
    # Sample y ~ N(mu, sigma^2)
    y = mu + sigma_safe * z
    
    # We want to minimize the objective function.
    # Therefore, a lower sampled value y is better.
    # The acquisition score should be higher for better candidates.
    # So, score = -y
    scores = -y
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return scores
        
    return scores
