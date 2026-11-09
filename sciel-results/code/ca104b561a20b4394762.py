import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return EI-based score
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Structural Change: Use Upper Confidence Bound (UCB)
    # UCB score = -mu + beta * sigma
    # We want to minimize mu - beta * sigma, so we return the negative for "higher is better"
    # Or simply return beta * sigma - mu
    
    # Choose beta. 
    # A common choice is beta = 2.0 or adaptive.
    # Let's use a fixed beta = 2.0 for simplicity and robustness.
    # This is a structural choice of the acquisition function family.
    beta = 2.0
    
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
