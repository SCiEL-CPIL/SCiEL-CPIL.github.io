"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function combining EI, PSI, and cost efficiency."""
    
    # Safety margin: if we're close to deadline, return simple EI/cost
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Significant Improvement (PSI)
    # P(f(x) < f_best - delta) where delta scales with f_best
    # When f_best is large, use larger delta; when close to 0, use smaller
    delta = max(0.01, 0.05 * f_best) if f_best > 0 else 0.01
    z_psi = (f_best - delta - mu) / sigma_safe
    psi = norm.cdf(z_psi)
    
    # Blend EI and PSI based on how close f_best is to 0
    # When f_best is large, weight PSI more (explore for big wins)
    # When f_best is small, weight EI more (exploit near optimum)
    if f_best > 1.0:
        w_psi = 0.5
    elif f_best > 0.1:
        w_psi = 0.3
    else:
        w_psi = 0.1
    
    combined = w_psi * psi + (1 - w_psi) * ei
    
    # Cost efficiency: prefer cheaper evaluations
    # Normalize cost: cost ranges from ~1 to ~2
    # Use inverse cost as weight, but don't over-penalize expensive points
    cost_norm = cost / np.mean(cost)  # normalize around 1
    cost_weight = 1.0 / np.maximum(cost_norm, 0.5)
    
    scores = combined * cost_weight
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
