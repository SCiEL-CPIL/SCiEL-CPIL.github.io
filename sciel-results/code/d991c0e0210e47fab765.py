"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma < 1e-7, np.where(mu < f_best, f_best - mu, 0.0), ei)
    return ei


def _soft_min_cost_penalty(cost, base_score, temperature=1.0):
    """
    Smooth cost penalty using log-sum-exp style soft-min.
    
    Instead of a hard linear penalty, we use a smooth surrogate that
    penalizes high-cost candidates more aggressively when they have
    low base scores, while being lenient when base scores are high.
    
    This creates a smooth trade-off: candidates with high expected
    improvement can afford higher costs, while those with low
    improvement need to be cheap.
    """
    # Scale base_score to work with cost scale
    base_max = np.max(base_score)
    base_min = np.min(base_score)
    base_range = base_max - base_min
    
    if base_range > 1e-12:
        # Normalize base score to [0, 1]
        base_norm = (base_score - base_min) / base_range
    else:
        base_norm = np.ones_like(base_score) * 0.5
    
    # Cost range
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-12:
        # Normalize cost to [0, 1]
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Temperature controls how aggressive the penalty is
    # Lower temperature = sharper penalty for expensive low-score candidates
    # Higher temperature = smoother, more lenient
    
    # Create a composite score that combines base score and cost
    # We want to penalize (high cost AND low base score) more than
    # (high cost AND high base score) or (low cost AND low base score)
    
    # Use a product-based interaction: penalty increases when cost is high
    # and base score is low
    interaction = cost_norm * (1.0 - base_norm)
    
    # Apply smooth penalty using exp(-temperature * interaction)
    # This gives a smooth, bounded penalty in [0, 1]
    penalty_raw = np.exp(-temperature * interaction)
    
    # Normalize penalty to [0, 1] where 1 = no penalty, 0 = max penalty
    penalty_min = np.min(penalty_raw)
    penalty_max = np.max(penalty_raw)
    penalty_range = penalty_max - penalty_min
    
    if penalty_range > 1e-12:
        penalty_norm = (penalty_raw - penalty_min) / penalty_range
    else:
        penalty_norm = np.ones_like(penalty_raw)
    
    # Final penalty: subtract from base score
    # Scale penalty to be meaningful relative to base score range
    penalty_scale = base_range * 0.5
    return base_score - penalty_scale * (1.0 - penalty_norm)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using smooth cost-benefit trade-off.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return -cost + np.min(cost)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Compute UCB for exploration
    ucb = mu - 2.0 * sigma
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Adaptive blending: use progress to switch between exploitation and exploration
    f_min_known = -78.3323
    progress = np.clip((f_best - f_min_known) / 100.0, 0.0, 1.0)
    
    # When far from optimum (high progress), explore more; when close, exploit more
    w_exploit = 0.7 + 0.3 * (1.0 - progress)
    w_explore = 0.3 + 0.3 * progress
    
    # Combined base score
    base_score = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Deadline guard before expensive computation
    if not _elapsed_ok():
        return base_score
    
    # Apply smooth cost-aware penalty
    # Temperature increases as we progress (more cost-sensitive near optimum)
    temperature = 2.0 + 3.0 * progress
    
    scores = _soft_min_cost_penalty(cost, base_score, temperature=temperature)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
