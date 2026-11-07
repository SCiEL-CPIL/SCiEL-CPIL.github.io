import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with adaptive xi.
    Uses a small xi to balance exploration and exploitation.
    """
    _ensure_start_time()
    
    # Adaptive xi: start with a larger xi for exploration, decrease over time
    # We don't have explicit step count, so use a fixed small xi
    xi = 0.01
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu - xi) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
