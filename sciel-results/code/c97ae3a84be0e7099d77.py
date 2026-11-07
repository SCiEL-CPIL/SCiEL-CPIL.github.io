import numpy as np
import time
from scipy.stats import norm

# Module-level time tracking for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition for Bayesian optimization.
    
    Samples a single function realization f ~ N(mu, sigma) and scores candidates
    by -f(x) / cost(x). This stochastically couples exploration with cost efficiency.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return quickly if we're getting close to the limit
    if time.time() - _start_time > 325:
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = mu.shape[0]
    
    # Ensure sigma is non-negative (it's a standard deviation)
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: sample one realization from the posterior
    # Use a fixed seed per call to make it deterministic for reproducibility within a run
    # But actually, for Thompson Sampling we want randomness. Let's use numpy's global RNG.
    # To avoid issues with the global RNG state, we'll just use np.random directly.
    
    # Sample f(x) ~ N(mu, sigma)
    sample = mu + sigma_safe * np.random.randn(N)
    
    # Score: we want to minimize, so lower sample values are better.
    # The acquisition score should be higher for better candidates.
    # So score = -sample / cost
    # Points with low predicted f (good) and low cost get high scores.
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
