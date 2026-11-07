"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition on reward (negative expected regret).
    
    Treats optimization as a budget-constrained multi-armed bandit:
    - Reward = -mu (expected negative objective, higher is better)
    - UCB on reward: -mu + beta * sigma
    - Cost-aware: divide by sqrt(cost) to prefer cheaper evaluations
    - Dynamic beta: scaled by remaining budget ratio to shift from
      exploration (early) to exploitation (late)
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return np.zeros(len(mu), dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate remaining budget ratio:
    # Total budget is 30, we've spent some cost. Use f_best and sigma stats
    # as proxies for progress. Early: high exploration, Late: high exploitation.
    # Use normalized position: if f_best is close to 0 (min), we're late.
    # Three hump camel min is 0, max is ~5.5 in [-5,5]^2
    f_scale = 5.5  # approximate range of three hump camel 2D
    progress = np.clip(1.0 - f_best / f_scale, 0.0, 1.0)
    remaining_ratio = 1.0 - progress
    
    # Dynamic beta: start with high exploration, decrease toward exploitation
    # beta ranges from ~2.0 (early) to ~0.3 (late)
    beta = 0.3 + 1.7 * remaining_ratio
    
    # UCB on reward: -mu + beta * sigma
    # Higher score = better expected reward with uncertainty bonus
    ucb_reward = -mu + beta * sigma_safe
    
    # Cost-aware scaling: divide by sqrt(cost)
    # This is a softer penalty than 1/cost, balancing exploration vs cost
    cost_factor = np.sqrt(cost_safe)
    score = ucb_reward / cost_factor
    
    # Normalize to ensure numerical stability
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
