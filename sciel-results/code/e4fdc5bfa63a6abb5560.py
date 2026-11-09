"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per the contract


def _check_deadline():
    """Return True if we are past the deadline (with a small safety margin)."""
    return (time.time() - _START_TIME) > 295.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
        mu: (2000,) array of GP mean predictions in objective units.
        sigma: (2000,) array of GP std deviations in objective units.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of evaluation costs (1 + mean of scaled coords).
    
    Returns:
        (2000,) array of finite scores; higher means more desirable to evaluate.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        # Fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    # Adaptive exploration coefficient that decays with budget consumed.
    # cost array values are in [1, 2] (1 + mean of two coords in [0,1]).
    # Total budget is 30. Initial 5 uniform points cost ~5*1.5 = 7.5.
    # So remaining budget ~ 22.5. As more evaluations happen, f_best improves
    # and we should exploit more.
    
    # Use a decay based on how much of the budget has been consumed.
    # cost[i] is the cost of evaluating candidate i. The total cost spent so far
    # is not directly available, but we can infer it from f_best and the
    # relationship. Instead, use a simple heuristic: decay beta with the
    # number of evaluations, which correlates with total cost.
    #
    # Alternative: use the magnitude of f_best as a proxy for progress.
    # Early on f_best is large (levy 2d can be ~100+), late it's near 0.
    # Normalize: beta starts high, decays as f_best approaches 0.
    
    # Levy 2d range: min is 0, typical initial values ~50-100.
    # Use a smooth decay: beta = beta_max * (1 + f_best / scale)^(-gamma)
    # When f_best is large, beta is large (explore); when f_best is small, beta is small (exploit).
    
    # Normalize f_best to [0, 1] range assuming max ~100
    f_best_clamped = np.clip(f_best, 0.0, 100.0)
    progress = f_best_clamped / 100.0  # 1.0 at start, 0.0 at optimum
    
    # Adaptive beta: starts at ~3.0, decays to ~0.5
    beta = 0.5 + 2.5 * progress
    
    # Cost-aware LCB: mu - beta * sigma, penalized by cost
    # We want to maximize: (mu - beta * sigma) / cost
    # But also incorporate a small exploration bonus for low-cost, high-uncertainty points
    
    # Compute the base LCB score
    lcb = mu - beta * sigma
    
    # Normalize cost to [0, 1] for weighting (cost is in [1, 2])
    cost_normalized = (cost - 1.0) / 1.0  # in [0, 1]
    
    # Cost-aware score: divide LCB by cost to prefer cheap evaluations
    # Add a small exploration term for high sigma at low cost
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    exploration_bonus = 0.1 * sigma_norm * (1.0 - cost_normalized)
    
    score = lcb / (cost + 1e-9) + exploration_bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
