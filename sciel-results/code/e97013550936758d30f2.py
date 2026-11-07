import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = 0.5  # 0.5 second limit per call, plenty for N=2000
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB with exploration-exploitation balance
    # Higher exploration early on (large f_best), more exploitation later (small f_best)
    # The Beale function has a global minimum of 0 at (3, 0.5) and local minima.
    # With only 30 evaluations, we need to balance finding the global basin vs. refining
    
    # Adaptive beta: starts around 5.0 when f_best is large, decreases as f_best approaches 0
    # This encourages broader exploration early, then focuses on exploitation
    beta = 5.0 * (f_best / (f_best + 1.0)) + 1.0
    
    # Compute UCB scores
    scores = -mu + beta * sigma_safe
    
    # Check if computation took too long (unlikely for simple arithmetic)
    if time.time() - start > limit:
        # Fallback to EI if something goes wrong
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
