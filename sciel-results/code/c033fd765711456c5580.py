"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf, erfinv

_DEADLINE_START = time.time()
_TIME_LIMIT = 295.0  # seconds, safety margin before 330s kill


def _norm_cdf(x):
    """Fast standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Structural Change: Cost-aware Expected Improvement with adaptive exploration.
    
    Uses EI (Expected Improvement) as the primary signal, normalized by cost.
    Adds a small uncertainty bonus to encourage exploration in early stages.
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    Score = EI / cost + beta * sigma / cost
    where beta adapts based on how close f_best is to the theoretical minimum (0).
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
        score = ei / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration weight:
    # When f_best is close to 0 (the minimum), reduce exploration
    # When f_best is large, increase exploration to find better regions
    # f_best should be >= 0 for Levy function
    # Use a sigmoid-like scaling: beta is high when f_best is large, low when f_best ~ 0
    f_best_safe = max(f_best, 1e-9)
    # beta ranges from ~0.01 (when f_best ~ 0) to ~0.5 (when f_best ~ 10+)
    beta = 0.5 / (1.0 + f_best_safe)
    
    # Uncertainty bonus
    uncertainty_bonus = beta * sigma_safe
    
    # Combined score: EI + uncertainty bonus, all normalized by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = (ei + uncertainty_bonus) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
