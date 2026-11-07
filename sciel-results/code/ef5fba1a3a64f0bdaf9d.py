import numpy as np
import time
from scipy.stats import norm

_module_start = time.time()
_DEADLINE = 275.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Compute Probability of Improvement (POI)
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    poi = norm.cdf(z)
    
    # Add a small exploration term based on uncertainty (sigma)
    # This helps escape local minima where POI might be low but uncertainty is high
    # We scale sigma to be comparable to POI (which is in [0, 1])
    # A simple scaling: 0.1 * sigma
    # If sigma is large, this term dominates, encouraging exploration
    # If sigma is small, POI dominates, encouraging exploitation
    
    # To make the exploration term more robust, we can normalize sigma
    # But simple addition is often effective
    exploration_weight = 0.1
    scores = poi + exploration_weight * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
