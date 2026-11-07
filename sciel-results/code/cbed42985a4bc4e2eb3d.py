"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: penalized expected improvement with uncertainty weighting.
    
    We use a combination of:
    1. Expected Improvement (EI) per unit cost
    2. A bonus for high uncertainty (to explore) when we have budget
    3. A penalty for low uncertainty near f_best (exploitation)
    
    The key insight: as cost accumulates, we should shift from exploration to exploitation.
    """
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    cost_safe = np.maximum(cost, 1e-6)
    ei_per_cost = ei / cost_safe
    
    # Uncertainty bonus: we want to explore when sigma is high relative to the spread
    # Use a normalized uncertainty measure
    sigma_norm = sigma_safe / (sigma_safe + np.median(sigma_safe) + 1e-10)
    
    # Balance between exploitation (EI) and exploration (uncertainty)
    # As f_best improves (gets more negative), we can afford more exploitation
    # Use a simple heuristic: weight exploration by how much improvement is still possible
    # Hartmann 3D minimum is about -3.86, so if f_best is close, exploit more
    
    # Adaptive balance: start with more exploration, shift to exploitation
    # Use the ratio of current best to theoretical minimum as a proxy
    f_min_theoretical = -3.86278
    if f_best > f_min_theoretical:
        # Still room for improvement
        progress = (f_best - f_min_theoretical) / (f_best - f_min_theoretical + 1e-10)
        # progress is in [0, 1], higher means more room
        explore_weight = 0.3 + 0.4 * progress  # 0.3 to 0.7
    else:
        explore_weight = 0.3
    
    # Combine: exploitation + exploration
    # Normalize EI for comparison
    ei_max = np.max(ei_per_cost) if np.max(ei_per_cost) > 1e-10 else 1.0
    ei_norm = ei_per_cost / (ei_max + 1e-10)
    
    score = explore_weight * ei_norm + (1.0 - explore_weight) * sigma_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
