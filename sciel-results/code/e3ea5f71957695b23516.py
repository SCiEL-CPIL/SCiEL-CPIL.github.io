import numpy as np
import time
from scipy.stats import norm

# Deadline guard
TIME_BUDGET = 310.0
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Returns scores to be maximized.
    We draw a single sample from the posterior at each candidate:
    f_s = mu + sigma * xi, where xi ~ N(0, 1)
    Then we return -f_s as the score (since lower f is better, we maximize -f).
    
    Parameters:
    mu: array of shape (N,), posterior mean in objective units.
    sigma: array of shape (N,), posterior standard deviation in objective units.
    f_best: float, current best observed value.
    
    Returns:
    array of shape (N,), finite scores.
    """
    # Deadline guard
    if _elapsed() > TIME_BUDGET:
        # Fallback: Simple LCB with small beta
        beta = 0.5
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = beta * sigma_safe - mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Thompson Sampling: draw a single random sample from the posterior
    # f_s = mu + sigma * xi, where xi ~ N(0, 1)
    # We want to minimize f, so we maximize -f_s
    xi = np.random.normal(0.0, 1.0, size=mu_safe.shape)
    f_sample = mu_safe + sigma_safe * xi
    
    # Score is negative of the sample (higher score = lower predicted f)
    scores = -f_sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
