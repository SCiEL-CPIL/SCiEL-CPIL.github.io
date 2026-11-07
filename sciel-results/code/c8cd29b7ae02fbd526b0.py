"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = None
_DEADLINE_LIMIT = 310.0  # Safety margin before 330s kill

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    _init_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # LCB (Lower Confidence Bound) for minimization:
    # We want to minimize, so lower mu - kappa*sigma is better.
    # Higher score = more desirable to evaluate.
    # So score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    # 
    # But we also want to incorporate f_best: points that are likely to beat f_best
    # should be preferred. Use a hybrid: LCB weighted by proximity to f_best.
    #
    # Structural change: Use a two-component score:
    # 1. LCB component: encourages low predicted value with uncertainty bonus
    # 2. Improvement component: EI-like term that rewards points likely to improve
    #
    # The LCB is fundamentally different from EI: it's a confidence-bound method
    # that doesn't require f_best in the core formula (only for weighting).
    
    kappa = 2.0  # Exploration-exploitation tradeoff
    
    # LCB score: lower (mu - kappa*sigma) is better for minimization
    lcb = mu - kappa * sigma_safe
    lcb_score = -lcb  # Negate so higher is better
    
    # EI component for additional signal
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = np.maximum(diff * cdf + sigma_safe * pdf, 0.0)
    
    # Normalize both components to [0, 1] roughly for combination
    # LCB range: mu can be anywhere, sigma is bounded by GP
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + 1e-12
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    ei_max = np.max(ei) + 1e-12
    ei_norm = ei / ei_max
    
    # Combine: 60% LCB, 40% EI
    combined = 0.6 * lcb_norm + 0.4 * ei_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if _check_deadline():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
