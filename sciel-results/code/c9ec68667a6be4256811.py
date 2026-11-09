"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses Thompson Sampling: samples from the posterior predictive distribution
    and selects the candidate with the minimum sampled value.
    Returns negative sampled values so that higher scores correspond to better candidates.
    """
    # Check time budget - if we're close to timeout, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return a simple LCB as fallback
        return -mu + 2.0 * sigma
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Generate standard normal random variables
    # Using a fixed seed is not allowed/possible per call context easily without global state,
    # but numpy.random is fine.
    z = np.random.randn(n)
    
    # Sample from the posterior: f_sample ~ N(mu, sigma^2)
    f_sample = mu + sigma * z
    
    # We want to minimize f_sample.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return -f_sample.
    scores = -f_sample
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
