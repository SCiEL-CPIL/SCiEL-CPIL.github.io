import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SAFETY = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-optimal cost-normalized Expected Improvement.
    
    Strategy:
    1. Compute standard Expected Improvement (EI).
    2. Normalize by evaluation cost (cost-aware).
    3. Compute a 'crowding' penalty based on local GP confidence.
       Regions with low sigma are considered 'well-explored' or 'dense'.
       We penalize candidates that fall into these dense, low-uncertainty regions
       to encourage the search to move to unexplored areas (high sigma) or
       areas with high potential improvement but less redundancy.
    4. The penalty is scaled to be significant enough to break local optima
       in the Levy function's many local minima.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    
    # Safety guard: if we are close to the deadline, return a simple, fast score
    if elapsed > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # 1. Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Avoid numerical issues in norm.cdf/pdf for extreme z
    z_clipped = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-normalize
    cost_safe = np.maximum(cost, 1e-9)
    
    # Pareto-optimal cost-normalization:
    # Use the minimum cost as the baseline (marginal cost relative to best cost)
    # This explicitly favors low-cost points that offer high expected improvement per unit of additional expense
    min_cost = np.min(cost_safe)
    marginal_cost = cost_safe - min_cost + 1e-9  # Ensure positive denominator
    
    # The score is EI divided by the marginal cost, which treats the search as a multi-objective optimization
    # where we want to maximize improvement while minimizing additional cost
    base_score = ei / marginal_cost
    
    # 3. Diversity/Crowding Penalty
    # Normalize sigma to [0, 1] for scaling
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max > sigma_min:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Penalty is high where sigma_norm is low (dense/explored)
    max_base = np.max(base_score)
    if max_base > 1e-9:
        penalty = 0.5 * max_base * (1.0 - sigma_norm) ** 2
    else:
        penalty = np.zeros_like(base_score)
    
    # 4. Combine
    score = base_score - penalty
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
