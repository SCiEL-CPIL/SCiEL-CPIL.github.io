"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Blended cost-aware acquisition: CEI + PoI + intrinsic exploration bonus.
    
    Structural change from pure CEI: combines multiple acquisition families
    with adaptive weighting and a curiosity-driven exploration term to handle
    the multimodal rastrigin landscape more effectively.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Z-score for EI/PoI
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-effective versions
    cei = ei / cost
    cpoi = poi / cost
    
    # Intrinsic exploration bonus: reward points with high relative uncertainty
    # This encourages visiting unexplored regions (important for rastrigin's many minima)
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    # Normalize sigma to [0, 1] range based on distribution
    sigma_rel = (sigma - np.min(sigma)) / (np.max(sigma) - np.min(sigma) + 1e-9)
    # Intrinsic bonus: high sigma relative to cost
    intrinsic = sigma_rel / cost
    
    # Adaptive blend weight: shift from exploration to exploitation as f_best improves
    # f_best starts high (~20 for rastrigin) and decreases toward 0
    # When f_best is high, explore more; when low, exploit more
    # Normalize f_best to [0, 1] assuming rastrigin range [0, ~20]
    f_best_norm = np.clip(f_best / 20.0, 0.0, 1.0)
    # Exploration weight: higher when f_best is worse (higher)
    explore_weight = 0.3 + 0.4 * f_best_norm  # range [0.3, 0.7]
    exploit_weight = 1.0 - explore_weight
    
    # Blend: weighted combination of CEI, CPoI, and intrinsic
    # CEI and CPoI are in different units, so normalize them
    cei_max = np.max(ceil) if np.max(ceil) > 1e-9 else 1.0
    cpoi_max = np.max(cpoi) if np.max(cpoi) > 1e-9 else 1.0
    intrinsic_max = np.max(intrinsic) if np.max(intrinsic) > 1e-9 else 1.0
    
    cei_norm = ceil / cei_max
    cpoi_norm = cpoi / cpoi_max
    intrinsic_norm = intrinsic / intrinsic_max
    
    # Final blended score
    score = (exploit_weight * (0.6 * cei_norm + 0.4 * cpoi_norm) +
             explore_weight * intrinsic_norm)
    
    # Scale back to reasonable magnitude
    score = score * (exploit_weight * cei_max + explore_weight * intrinsic_max)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
