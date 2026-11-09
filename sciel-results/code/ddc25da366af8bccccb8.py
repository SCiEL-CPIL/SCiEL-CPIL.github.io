import numpy as np
import time

_start_time = time.time()

def _get_elapsed_time():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling proxy heuristic.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float, lowest observed value
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling proxy: (f_best - mu) / sigma
    # This normalizes the potential improvement by the uncertainty,
    # encouraging exploration of points with high potential relative to their noise.
    score = (f_best - mu) / sigma
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
