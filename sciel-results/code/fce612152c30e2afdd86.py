import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best):
    """
    Pure GP-LCB (Lower Confidence Bound) acquisition for minimization.
    Score = -mu - kappa * sigma, so higher scores prefer low mean and high uncertainty.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Deadline guard: return a simple exploitation score if we're running out of time
    if time.time() > _DEADLINE:
        return -mu

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Pure GP-LCB: minimize mu + kappa * sigma, so score = -(mu + kappa * sigma)
    # Fixed exploration parameter
    kappa = 2.0
    
    scores = -mu - kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
