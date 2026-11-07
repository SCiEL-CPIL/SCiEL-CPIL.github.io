import numpy as np
import time

# Capture start time when the module is loaded
_START_TIME = time.time()
_TIMEOUT_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Returns scores where higher is better (i.e., lower predicted function value).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT - _SAFETY_MARGIN:
        # If we are close to the deadline, fall back to a fast, deterministic greedy approach
        # to ensure we return *something* valid quickly.
        return -mu

    # Thompson Sampling:
    # 1. Sample from the posterior predictive distribution: f(x) ~ N(mu(x), sigma(x)^2)
    # 2. The score should favor candidates with low sampled values (since we minimize).
    # 3. Score = -f_sampled
    
    # Generate standard normal samples
    # Using np.random.normal is fast for N=2000
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    
    # Compute sampled function values
    f_samples = mu + sigma * z
    
    # Convert to scores (higher is better)
    scores = -f_samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
