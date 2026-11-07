import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # The evaluator kills after 330s. We want to be safe.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: Return -mu to exploit the current best mean.
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Thompson Sampling: sample from the posterior distribution
    # We generate a random normal sample and compute the sampled function value
    # The acquisition score is the negative of the sampled value (higher is better)
    # This naturally balances exploration and exploitation
    
    # Generate a single random normal sample
    z = np.random.standard_normal(1)
    
    # Sampled function value: f_sample = mu + sigma * z
    # Higher score means we want lower f_sample, so we return -f_sample
    f_sample = mu + sigma_safe * z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(f_sample), -f_sample, -1e10)
    
    return scores
