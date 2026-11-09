import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    # Deadline guard
    # The computation is fast, but we check just in case.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Epsilon to prevent issues with zero/negative sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling:
    # Sample from the posterior: y ~ N(mu, sigma)
    # We want to minimize f, so we prefer low y.
    # Score should be higher for lower y.
    # score = -y
    z = np.random.randn(len(mu))
    samples = mu + sigma_safe * z
    scores = -samples
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() - start_time > 320:
         # If somehow slow, return a fallback
         # But this shouldn't happen
         pass
         
    return scores
