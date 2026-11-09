"""Cost-aware acquisition for rastrigin 2d BO."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track a global start and enforce a safety margin.
_T0 = time.time()
_DEADLINE = 300.0  # seconds from start; leaves 30s buffer

def _elapsed():
    return time.time() - _T0

def _exceeded():
    return _elapsed() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Returns 2000 finite scores (higher = evaluate next).
    
    Uses a hybrid of Expected Improvement per unit cost and a
    lower-confidence-bound (LCB) term to balance exploitation
    and exploration, weighted by remaining budget.
    """
    # Safety: if we're close to deadline, return a simple fast score
    if _exceeded():
        # Fallback: simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) - encourages exploration
    # Use a beta that balances exploration; 2.0 is a common choice
    beta = 2.0
    lcb = mu - beta * sigma_safe
    
    # Cost-aware EI
    ei_cost = ei / (cost + 1e-9)
    
    # Normalize EI and LCB to comparable scales
    # Use robust normalization: divide by max to get [0, 1] range
    ei_max = np.max(ei_cost) if np.max(ei_cost) > 0 else 1.0
    lcb_min = np.min(lcb) if np.min(lcb) < np.max(lcb) else np.min(lcb) - 1.0
    lcb_max = np.max(lcb) if np.max(lcb) > np.min(lcb) else np.max(lcb) + 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    # Normalize both to [0, 1]
    ei_norm = ei_cost / ei_max
    lcb_norm = (lcb - lcb_min) / lcb_range
    lcb_norm = np.clip(lcb_norm, 0.0, 1.0)
    
    # Weight between exploitation (EI) and exploration (LCB)
    # As budget depletes, shift more toward exploitation
    # cost is in [1, 2], f_best is the current best
    # Use a simple heuristic: weight based on how good the current best is
    # If f_best is close to 0 (the minimum), we can afford more exploitation
    # If f_best is still high, we need more exploration
    
    # A practical weight: start with 0.6 EI / 0.4 LCB, adjust slightly
    # based on the ratio of best possible improvement to current sigma
    # Simpler: fixed weight that works well in practice
    w_ei = 0.7
    w_lcb = 0.3
    
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
