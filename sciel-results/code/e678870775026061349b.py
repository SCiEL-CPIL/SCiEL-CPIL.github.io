import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # UCB parameter
    beta = 2.0
    
    # Sanitize inputs to ensure finite outputs
    # Replace NaN/Inf in mu and sigma with neutral or extreme values
    # For minimization, high mu is bad, high sigma is good (exploration).
    # If mu is NaN, assume it's bad (high). If sigma is NaN, assume it's 0 (no info).
    # However, simply masking non-finite scores at the end is robust.
    
    mu_clean = np.where(np.isfinite(mu), mu, 1e6) # High cost if unknown
    sigma_clean = np.where(np.isfinite(sigma), sigma, 0.0) # No uncertainty if unknown
    
    sigma_safe = np.maximum(sigma_clean, 1e-8)
    
    scores = -mu_clean + beta * sigma_safe
    
    # Ensure final scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    elapsed = time.time() - start_time
    if elapsed > 250:
        pass # Calculation is fast, this is a formality
        
    return scores
