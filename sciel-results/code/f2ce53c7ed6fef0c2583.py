import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB for minimization
    # Score = -mu + beta * sigma
    # Higher score is better
    
    # Estimate exploration strength based on current best
    # Global min is 3. 
    # Assume initial f_best is around 20-30.
    # Scale beta from 0 to 2.5 (or higher) as f_best goes from 3 to 28.
    
    # Clip f_best to avoid negative beta if f_best < 3 (shouldn't happen, but safe)
    f_best_clipped = max(f_best, 3.0)
    
    # Define the range over which we adapt
    f_min = 3.0
    f_max_ref = 28.0 # Approximate max value of Goldstein-Price in domain
    
    # Normalized progress (0 at min, 1 at max)
    # We want beta to be high when f_best is high (early) and low when f_best is low (late)
    # So beta proportional to (f_best - f_min)
    
    span = f_max_ref - f_min
    if span == 0:
        beta = 0.0
    else:
        # Clamp ratio to [0, 1] or allow >1?
        # If f_best > f_max_ref, beta > 2.5. That's fine.
        ratio = (f_best_clipped - f_min) / span
        # Base beta 2.5
        beta = 2.5 * ratio
        
    # Calculate score
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma_safe
    
    # Return finite scores
    # If any non-finite, replace with very low value (bad score)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
