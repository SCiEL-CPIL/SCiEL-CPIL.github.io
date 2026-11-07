"""Cost-aware Thompson Sampling acquisition with variance penalty for stability."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Uses a variance penalty to stabilize GP predictions in high dimensions,
    and models the cost-benefit tradeoff via expected improvement per unit cost.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return deterministic fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Variance penalty: add a small floor to sigma to stabilize in high-d
    # This reduces sensitivity to noise and improves consistent exploration
    sigma_penalized = sigma_safe + 0.01
    
    z = (f_best - mu) / sigma_penalized
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_penalized * norm.pdf(z)
    
    # Cost-aware: expected improvement per unit cost
    # Higher EI and lower cost both increase the score
    score = ei / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in high-dimensional space
    exploration_bonus = 0.1 * sigma_penalized / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
