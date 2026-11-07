import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of Expected Improvement and a diversity/exploration term,
    scaled by cost.
    """
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Expected Improvement (EI)
    # Use a small xi to balance exploration and exploitation
    # xi scales with f_best to encourage exploration early on
    xi = 0.01 * max(f_best, 0.0)
    z = (f_best - xi - mu) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Diversity term: use sigma as a proxy for information gain
    # In regions with high uncertainty (high sigma), we want to explore more
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > eps else 1.0
    diversity = sigma_safe / sigma_max
    
    # Adaptive blending: 
    # Early on (high f_best), favor exploration/diversity
    # Later (low f_best), favor exploitation via EI
    # f_best for griewank is in [0, ~0.3], so use it as exploration weight
    # Clip to ensure we always have some exploration
    explore_weight = np.clip(f_best / 0.05, 0.2, 1.0)
    exploit_weight = 1.0 - explore_weight
    
    # Normalize EI to [0, 1] for comparable scaling with diversity
    ei_max = np.max(ei) if np.max(ei) > eps else 1.0
    ei_norm = ei / ei_max
    
    # Combined acquisition: blend EI and diversity
    raw_score = exploit_weight * ei_norm + explore_weight * diversity
    
    # Cost-aware: divide by cost to penalize expensive points
    # Use a soft penalty: 1 / (1 + cost) instead of 1/cost to avoid extreme penalties
    cost_penalty = 1.0 / (1.0 + cost_safe)
    score = raw_score * cost_penalty
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
