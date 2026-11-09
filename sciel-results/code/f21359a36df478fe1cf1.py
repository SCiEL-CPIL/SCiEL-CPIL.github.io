"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 325.0  # 5 seconds margin before 330s

def acquisition(mu, sigma, f_best):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE:
        # Return a simple greedy score to avoid timeout
        return -mu.copy()
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter
    # As we get closer to the optimum, we should exploit more
    # Use a simple heuristic: start with high exploration, decrease over time
    # We can estimate progress by how close f_best is to 0
    # Weierstrass minimum is 0, so f_best should approach 0
    
    # For Weierstrass 1D on [-0.5, 0.5], typical values range from 0 to ~1.5
    # Start with beta=2.0 (high exploration), decrease to ~0.5
    
    # Simple schedule based on f_best
    # If f_best is large (>0.5), we're early, explore more
    # If f_best is small (<0.1), we're close, exploit more
    
    if f_best > 0.5:
        beta = 2.0
    elif f_best > 0.2:
        beta = 1.5
    elif f_best > 0.1:
        beta = 1.0
    elif f_best > 0.05:
        beta = 0.7
    elif f_best > 0.01:
        beta = 0.5
    else:
        beta = 0.3
    
    # Lower Confidence Bound for minimization
    # Score = mu - beta * sigma
    # Higher score means more likely to be good (low f value)
    scores = mu - beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
