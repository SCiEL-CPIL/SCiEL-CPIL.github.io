"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track when the module was first called
_start_time = None
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s kill

def _check_deadline():
    """Return True if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _DEADLINE_SECONDS


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive POI + normalized EI + cost-aware UCB
    with additive cost penalty.
    
    Structural change: Uses Probability of Improvement (POI) as primary signal,
    normalized EI for scale invariance, and additive cost penalty instead of division.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # If we're near deadline, return a simple safe score
    if _check_deadline():
        return np.zeros(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # z-score for POI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Normalized EI: scale-invariant expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI by a reference scale (use max sigma as reference)
    ref_scale = np.maximum(np.max(sigma), eps)
    ei_norm = ei / ref_scale
    
    # Cost-aware UCB: uncertainty adjusted by cost
    # Use additive cost penalty instead of division for different correction mechanism
    ucb = mu + 1.5 * sigma_safe - 0.5 * cost_safe
    
    # Combine: POI (primary) + normalized EI (scale-invariant) + UCB (exploration)
    # Weight POI heavily as it directly measures probability of improvement
    score = 2.0 * poi + 1.0 * ei_norm - 0.3 * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
