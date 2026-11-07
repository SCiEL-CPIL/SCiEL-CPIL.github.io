import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s kill

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return elapsed < (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Pure Probability of Improvement (POI) acquisition function for minimization.
    
    POI scores each candidate by the probability that its true value is lower
    than the current best f_best. This provides a distinct, binary-like exploration
    signal that differs from the continuous expected improvement metric.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    if not _ensure_deadline():
        # Fallback: return POI scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        return np.where(np.isfinite(poi), poi, 0.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI) for minimization
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(poi), poi, 0.0)
    
    return scores
