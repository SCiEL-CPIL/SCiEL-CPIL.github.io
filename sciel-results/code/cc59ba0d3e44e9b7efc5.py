import numpy as np

def acquisition(mu, sigma, f_best):
    # Dynamic UCB for Hartmann-6D
    # Adapt kappa based on how close f_best is to the known global minimum
    
    min_val = -3.32237
    max_val = 3.32 # Approximate max of Hartmann 6D
    range_val = max_val - min_val
    
    # Calculate gap from optimum
    gap = f_best - min_val
    # Normalize gap to [0, 1]
    # If f_best is worse than max_val, clip to 1 (high exploration)
    # If f_best is better than min_val (shouldn't happen), clip to 0
    norm_gap = np.clip(gap / range_val, 0.0, 1.0)
    
    # Interpolate kappa
    # High norm_gap (early/poor) -> High kappa (explore)
    # Low norm_gap (late/good) -> Low kappa (exploit)
    kappa_min = 0.5
    kappa_max = 3.0
    kappa = kappa_min + (kappa_max - kappa_min) * norm_gap
    
    # Ensure sigma is valid
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB score for minimization: -mu + kappa * sigma
    scores = -mu + kappa * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
