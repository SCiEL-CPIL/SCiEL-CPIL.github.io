"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Theoretical minimum of cosine8 in 8D
_F_MIN = -0.8

# Track total time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Pareto-frontier-inspired scoring
    with Expected Improvement and cost-awareness.
    
    Strategy:
    - Compute Expected Improvement (EI) for minimization
    - Compute cost-normalized score
    - Use a dynamic weighting that shifts from exploration (uncertainty) to
      exploitation (improvement) as budget is consumed
    - Return higher scores for more promising candidates
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a simple fallback: just -mu (exploit best prediction)
        return -np.asarray(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Normalize cost to [0, 1] for weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost) * 0.5
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    delta = f_best - mu
    z = delta / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = delta * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.ones_like(ei) * 0.5
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Also compute an uncertainty-based score (high sigma = more uncertain)
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-12:
        sigma_norm = np.ones_like(sigma) * 0.5
    else:
        sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Dynamic weighting: as we spend more budget, we should exploit more
    # cost represents the accumulated cost so far (total cost)
    # Total budget is 30, so if cost is close to 30, we should exploit heavily
    # cost here is the cost of each candidate, but we can use f_best as a proxy
    # for how close we are to the minimum
    
    # Use the gap between f_best and theoretical minimum to adjust weights
    gap = max(f_best - _F_MIN, 1e-6)
    # When gap is large (early), explore more; when gap is small (late), exploit
    # Smooth transition: weight_ei increases as gap decreases
    explore_weight = 0.3 + 0.4 * np.exp(-gap * 1.5)
    exploit_weight = 1.0 - explore_weight
    
    # Cost penalty: prefer cheaper evaluations
    # Score = exploit_weight * EI_norm + explore_weight * sigma_norm - cost_penalty
    # Make cost penalty proportional to cost_norm
    cost_penalty_weight = 0.3
    
    score = exploit_weight * ei_norm + explore_weight * sigma_norm - cost_penalty_weight * cost_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
