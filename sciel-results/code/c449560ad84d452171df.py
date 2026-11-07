import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function for Minimization.
    Samples from the posterior for each candidate and selects the one
    with the lowest sampled value to balance exploration and exploitation.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback to greedy to ensure fast return
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Thompson Sampling: sample from N(mu, sigma) for each candidate
    # Use a fixed random state for reproducibility within the function call
    # Actually, we want randomness across calls for exploration
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma)
    
    # For minimization, higher score means more promising
    # We want the candidate with the LOWEST sampled value to have the HIGHEST score
    # So we negate the samples
    scores = -samples
    
    # Handle any remaining numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
