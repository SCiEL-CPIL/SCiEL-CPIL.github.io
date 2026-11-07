import numpy as np

def acquisition(mu, sigma, f_best):
    # Structural Change: Lower Confidence Bound (LCB)
    # Maximizes: beta * sigma - mu
    # This favors high uncertainty and low mean.
    
    # Safety guard for time (though not strictly necessary for vectorized ops)
    # import time
    # start = time.time()
    # ...
    # if time.time() - start > 290: return np.zeros_like(mu)
    
    # Use a fixed high exploration constant
    beta = 3.0
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate LCB score (higher is better)
    scores = beta * sigma_safe - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
