"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with adaptive exploration.
    
    Uses Expected Improvement normalized by cost, with an adaptive
    exploration bonus that scales with uncertainty and decreases as
    the best value approaches zero.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure positive values
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -20.0, 20.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost
    ei_cost = ei / cost_safe
    
    # Adaptive exploration: reward high uncertainty, but scale down
    # as we get closer to the optimum (f_best -> 0)
    # The exploration term should be significant early on but diminish
    # as we converge
    
    # Scale factor based on how close we are to the optimum
    # When f_best is large, we can afford more exploration
    # When f_best is small, we should be more exploitative
    exploration_scale = np.sqrt(max(abs(f_best), 1e-6))
    
    # UCB-style exploration: beta * sigma
    # Use a moderate beta that adapts to the scale
    beta = 0.8
    ucb_exploration = beta * sigma_safe * exploration_scale
    
    # Combine: primary exploitation (EI/cost) + secondary exploration
    # Normalize EI to be comparable to exploration term
    # Use the median EI as a reference scale
    ei_median = np.median(ei_cost) if np.any(ei_cost > 0) else 1.0
    ei_median = max(ei_median, 1e-6)
    
    # Weight the exploration term relative to EI
    # As f_best decreases, reduce exploration
    w_exploration = min(1.0, exploration_scale / max(np.std(mu) if np.std(mu) > 1e-6 else 1.0, 1e-6))
    w_exploration = 0.3 * w_exploration
    
    # Final score
    score = ei_cost + w_exploration * ucb_exploration / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard: if we're taking too long, return what we have
    elapsed = time.time() - t_start
    if elapsed > 5.0:  # 5 second safety margin per call
        pass  # We already have our result, just return it
    
    return score
# EVOLVE-BLOCK-END
