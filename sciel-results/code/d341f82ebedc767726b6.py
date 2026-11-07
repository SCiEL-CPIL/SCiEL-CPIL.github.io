"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses Expected Improvement (EI) normalized by cost for cost-awareness.
    EI directly targets improvement over f_best and is well-suited for
    minimization problems in Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Upper Confidence Bound (UCB) for minimization ---
    # Structural change: Use UCB instead of EI.
    # UCB_min(x) = mu(x) - beta * sigma(x)
    # We want to minimize UCB_min, so we maximize (f_best - UCB_min).
    # This explicitly rewards high uncertainty (sigma) via the beta term,
    # providing a different exploration/exploitation balance than EI.
    
    # Beta parameter: controls exploration. 
    # 1.0 is a reasonable default for moderate exploration.
    beta = 1.0
    
    ucb_min = mu - beta * sigma_safe
    
    # Potential improvement over f_best
    # If UCB_min is lower than f_best, we have potential improvement.
    # Score is the gap, penalized by cost.
    potential_improvement = f_best - ucb_min
    
    # Clip to non-negative to avoid rewarding points that are already known to be worse
    # (Though UCB_min can be > f_best, in which case potential_improvement is negative)
    potential_improvement = np.maximum(potential_improvement, 0.0)
    
    # Cost-awareness: divide by cost
    scores = potential_improvement / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
