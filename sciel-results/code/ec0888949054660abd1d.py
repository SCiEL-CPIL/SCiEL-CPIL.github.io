import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Upper Confidence Bound (UCB) with adaptive exploration parameter.
    For minimization, we want low mu and high sigma (uncertainty).
    UCB = mu - kappa * sigma (lower is better for the function, but we return higher scores)
    So score = -mu + kappa * sigma
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use -mu (pure exploitation)
        scores = -mu.copy()
        scores[~np.isfinite(scores)] = 0.0
        return scores
    
    # Handle cases where sigma is very small or zero
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive kappa: start with more exploration, decrease over time
    # Since we don't know the step number, use a heuristic based on f_best
    # As f_best gets closer to the minimum (-1.8013), we should explore less
    # The range of Michalewicz 2D is roughly [-1.8, 1.0]
    # Normalize f_best to get a progress estimate
    f_min_theoretical = -1.8013
    f_max_est = 1.0
    
    if f_max_est > f_min_theoretical:
        progress = (f_max_est - f_best) / (f_max_est - f_min_theoretical)
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = 0.5
    
    # Kappa decreases from ~2.0 to ~0.5 as we progress
    kappa = 2.0 - 1.5 * progress
    
    # UCB for minimization: we want to minimize mu - kappa*sigma
    # So the score (higher is better) is: -mu + kappa * sigma
    scores = -mu + kappa * safe_sigma
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
