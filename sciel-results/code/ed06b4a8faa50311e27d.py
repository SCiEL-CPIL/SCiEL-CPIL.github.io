import numpy as np
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 320s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to a simple deterministic score if we are running out of time
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost

    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Thompson Sampling:
    # Sample a single latent function value y from N(mu, sigma^2) for each candidate
    # y = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    y = mu + safe_sigma * z
    
    # Score: We want to minimize y, so higher score is better for lower y.
    # We also want to minimize cost, so we divide by cost.
    # Score = -y / cost
    score = -y / safe_cost
    
    # Ensure all values are finite
    # If any value is NaN or Inf, replace it with a very low number so it's not selected
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
