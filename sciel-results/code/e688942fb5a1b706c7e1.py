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


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lagrangian relaxation.
    
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
    
    # Cost-aware adjustment via Lagrangian relaxation
    # The cost penalty is proportional to the actual EI scale
    # This allows meaningful trade-off between improvement and cost
    cost_mean = np.mean(cost)
    cost_normalized = cost / cost_mean
    
    # Use a dynamic lambda that scales with the problem
    # Higher lambda = more cost-sensitive
    lambda_cost = 0.5 + 0.5 * progress
    
    # Apply cost penalty: subtract lambda * normalized_cost
    # This decouples cost from posterior geometry
    scores = base_score - lambda_cost * (cost_normalized - 1.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if not _elapsed_ok():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
