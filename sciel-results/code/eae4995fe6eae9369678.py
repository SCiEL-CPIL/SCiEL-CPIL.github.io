# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    # Deadline guard to ensure we don't exceed time limits
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        # Return a neutral score if we are running out of time
        # Using zeros is safe as it doesn't favor any point specifically,
        # but usually we want to return something that doesn't crash.
        # In practice, this should rarely trigger if the function is fast.
        return np.zeros_like(mu, dtype=float)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero or negative uncertainty
    # In LCB, we add beta * sigma, so we just need sigma to be non-negative.
    sigma_safe = np.maximum(sigma, 1e-9)

    # Strategy: Pure Lower Confidence Bound (LCB) for minimization
    # Score = -mu + beta * sigma
    # Higher score means we prefer lower mu (good) and higher sigma (exploration).
    # Based on Inspiration 1 (score 6.79), a high fixed beta works well for this task.
    # We use beta = 5.0 as it showed strong performance in escaping local minima
    # or poor initial regions in the Beale function optimization.
    
    beta = 5.0
    
    # Calculate the acquisition score
    score = -mu + beta * sigma_safe
    
    # Ensure the output is finite to prevent invalid results
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
