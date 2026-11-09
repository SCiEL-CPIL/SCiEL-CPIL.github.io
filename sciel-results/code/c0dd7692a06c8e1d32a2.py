import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety guard: if we are taking too long, return a simple result.
    # However, the computation is very fast (numpy vectorized).
    # The timeout is likely for the whole run or GP fitting (which we don't do).
    # We assume this function is fast enough.
    
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Ensure numerical stability
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = delta * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # Using subtraction instead of division to avoid over-penalizing expensive regions
    # when EI is high.
    # Cost range is [1, 2].
    # EI range depends on f_best and mu.
    # If f_best is close to 0, EI is small.
    # Let's use a small penalty.
    score = ei - 0.1 * cost
    
    # Add exploration bonus to encourage visiting high uncertainty areas
    # This helps escape local minima in the small budget scenario.
    score += 0.1 * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
