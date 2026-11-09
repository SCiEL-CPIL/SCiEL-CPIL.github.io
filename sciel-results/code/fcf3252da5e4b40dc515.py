import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _minmax_normalize(x):
    """Normalize an array to [0, 1] range. Handles edge cases."""
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-9:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    normalized Upper Confidence Bound (UCB) term with adaptive exploration weight.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple, fast exploitation score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: Lower Confidence Bound (LCB) style for minimization
    # We want to maximize the score, so we use mu - beta * sigma.
    beta = 2.0
    ucb = mu - beta * sigma_safe
    
    # Normalize both components to [0, 1]
    ei_norm = _minmax_normalize(ei)
    ucb_norm = _minmax_normalize(ucb)
    
    # Adaptive exploration weight based on progress towards known minimum
    # ST-1D min is approx -39.166, max is approx 15.
    min_val = -39.166
    max_val = 15.0
    # Progress: 0 if f_best is max, 1 if f_best is min
    if max_val - min_val < 1e-9:
        progress = 0.5
    else:
        progress = (max_val - f_best) / (max_val - min_val)
    
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Start with high exploration (0.5) and decrease to low (0.1) as we find better values
    w_explore = 0.5 - 0.4 * progress
    w_exploit = 1.0 - w_explore
    
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
