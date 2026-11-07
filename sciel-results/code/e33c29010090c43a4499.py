import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost exponent
    # If f_best is high, we are early, use linear cost penalty (alpha=1).
    # If f_best is low, we are late, use quadratic cost penalty (alpha=2).
    # Transition between 1 and 2 based on f_best.
    # Assume f_best drops from ~100 to ~0.
    # Use a log scale or linear clip.
    # Let's say if f_best > 1.0, alpha = 1.0.
    # If f_best < 0.0, alpha = 2.0.
    # Linear interpolation: alpha = 2.0 - f_best, clipped to [1.0, 2.0].
    
    # Softer dynamic cost exponent
    # Start with low penalty (alpha ~ 1.0) and increase slightly as we converge.
    # Max alpha is 1.5, avoiding the strong bias of alpha=2.0.
    alpha = 1.0 + 0.5 * np.exp(-np.maximum(f_best, 0.0))
    
    scores = ei / np.power(cost_safe, alpha)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
