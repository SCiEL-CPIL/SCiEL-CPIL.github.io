"""Direct cost-aware acquisition: normalize cost, use EI with robust sigma floor and optional UCB blend."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard (shared across calls in the loop)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per evaluation; use safety margin

def _elapsed_ok():
    """Return True if we have at least ~5s margin before the hard limit."""
    return (time.time() < _DEADLINE - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple score quickly
    if not _elapsed_ok():
        sigma_safe = np.maximum(sigma, 1e-6)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    # Handle cases where z is very large (EI ~ f_best - mu) or very negative (EI ~ 0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Make EI non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Normalize EI/cost to [0, 1] for combination
    ei_c_min = np.min(ei_cost)
    ei_c_max = np.max(ei_cost)
    ei_c_range = ei_c_max - ei_c_min
    if ei_c_range > 1e-12:
        ei_c_norm = (ei_cost - ei_c_min) / ei_c_range
    else:
        ei_c_norm = np.ones_like(ei_cost)
    
    # UCB: -mu + kappa * sigma (negative because we minimize)
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Weighted combination: 0.7 for cost-aware EI, 0.3 for UCB
    score = 0.7 * ei_c_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
