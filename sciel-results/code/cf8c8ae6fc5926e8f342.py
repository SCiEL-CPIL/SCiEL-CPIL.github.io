import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 320s to leave 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points.
    sigma (np.ndarray): Posterior standard deviation at candidate points.
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point. Higher score is better.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: use LCB with small kappa for speed
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    return scores
