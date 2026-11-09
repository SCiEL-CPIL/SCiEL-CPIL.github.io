import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 300s limit, keep 10s margin
    if time.time() - _start_time > 290:
        # Return a fallback: prioritize low cost and high uncertainty
        # This is a simple heuristic to ensure we return something valid
        # if we are running out of time
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        # Fallback score: high sigma, low cost
        scores = sigma / np.maximum(cost, 1e-9)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Use a sharper cost penalty (exponent 0.8)
    # This aligns with the "sharper decision boundary" finding which improved score.
    # It encourages focusing on high-EI/low-cost pockets more aggressively.
    cost_penalty = np.power(cost_safe, 0.8)
    
    scores = ei / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
