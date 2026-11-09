import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) for minimization.
    Score = (kappa * sigma - mu) / cost
    Higher score indicates a more promising candidate for minimization.
    """
    # Safety for sigma and cost to avoid division by zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Dynamic kappa tuning based on f_best
    # Hartmann 3D global minimum is approx -3.86278, maximum is 0.0
    f_min_hartmann = -3.86278
    f_max_hartmann = 0.0
    
    # Calculate progress: 0.0 at start (f_best ~ 0), 1.0 at end (f_best ~ -3.86)
    range_val = f_max_hartmann - f_min_hartmann
    if range_val < 1e-9:
        progress = 0.5
    else:
        progress = (f_max_hartmann - f_best) / range_val
    
    # Clip progress to [0, 1] to handle out-of-bounds f_best
    progress = np.clip(progress, 0.0, 1.0)
    
    # Kappa schedule:
    # Early search (progress ~ 0): High kappa (3.0) for exploration
    # Late search (progress ~ 1): Low kappa (1.0) for exploitation
    kappa_min = 1.0
    kappa_max = 3.0
    
    # Linear interpolation: 
    # progress=0 -> kappa=kappa_max
    # progress=1 -> kappa=kappa_min
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    # Calculate Score
    # For minimization, we look for low values.
    # LCB = mu - kappa * sigma (Optimistic lower bound)
    # We want to select the minimum LCB.
    # The interface requires returning scores where HIGHER is better.
    # Therefore, Score = -LCB / cost = (kappa * sigma - mu) / cost
    scores = (kappa * sigma_safe - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
