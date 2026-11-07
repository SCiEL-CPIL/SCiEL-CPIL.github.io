"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _safe_z(z):
    """Clamp z to avoid numerical issues in norm.cdf/pdf."""
    return np.clip(z, -10, 10)

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 330s limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 325:  # 5s safety margin
        return np.ones_like(mu) * 1.0
    
    # Expected improvement per unit cost (EI/cost)
    z = _safe_z((f_best - mu) / (sigma + 1e-9))
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    ei_cost = ei / (cost + 1e-9)
    
    # Add exploitation term: prefer low mu (high negative mu) relative to cost
    # This helps commit to promising low-mean regions
    exploit = -mu / (cost + 1e-9)
    
    # Add relative uncertainty term: prefer points with meaningful uncertainty
    # relative to the gap between f_best and mu
    sigma_gap = np.maximum(sigma, 1e-9)
    explore = sigma_gap / (np.abs(f_best - mu) + sigma_gap + 1e-9)
    
    # Combine: EI/cost dominates, with exploitation and exploration as secondary terms
    # Weights chosen to balance: EI is primary, exploitation helps commit, exploration prevents premature convergence
    score = ei_cost + 0.5 * exploit + 0.1 * explore
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
