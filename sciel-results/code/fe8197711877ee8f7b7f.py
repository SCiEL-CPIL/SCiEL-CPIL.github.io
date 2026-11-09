"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 2.0  # seconds to spare before the 330s wall-clock kill

def _ensure_deadline():
    """Set the deadline once on first call."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330.0 - _DEADLINE_MARGIN
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson-sampling-inspired cost-aware acquisition.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    deadline = _ensure_deadline()
    
    # Guard: if we're close to deadline, return a safe deterministic fallback
    if time.time() > deadline - 1.0:
        # Fallback: simple EI/cost, which is fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    n = len(mu)
    
    # Cost-aware Thompson sampling:
    # 1. Sample from GP posterior: f_sample = mu - sigma * z (minimization)
    #    Actually for minimization, we want low values to be "good".
    #    Thompson sample: f_tilde = mu - sigma * z where z ~ N(0,1)
    #    But we need to balance exploration vs exploitation with cost.
    
    # Strategy: Use a weighted combination of:
    # - Exploitation: how much better than f_best (in objective space)
    # - Exploration: uncertainty (sigma)
    # - Cost efficiency: divide by cost
    
    # Compute standard z-scores
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization: improvement = f_best - f)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) variant for minimization
    # UCB = mu - beta * sigma (we want low mu, high sigma for exploration)
    # Adaptive beta based on cost: cheaper points get more exploration
    beta = 0.5
    ucb = mu - beta * sigma_safe
    
    # Thompson sample: f_tilde = mu - sigma * z_sample
    # Use a deterministic proxy: the "median" Thompson sample is mu
    # but we can use a quantile to balance exploration/exploitation
    # Let's use the 25th percentile of the posterior (more exploratory)
    # f_25 = mu - 0.6745 * sigma (25th percentile of N(mu, sigma^2))
    f_quantile = mu - 0.6745 * sigma_safe
    
    # Combine EI and UCB with cost awareness
    # Normalize ei and ucb to [0, 1] range for stable weighting
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_range > 1e-12 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range  # lower ucb is better, so invert
    ucb_norm = 1.0 - ucb_norm  # now higher is better
    
    # Cost normalization: cost is in [1, 2] roughly
    cost_norm = cost / np.max(cost) if np.max(cost) > 1e-12 else 1.0
    cost_efficiency = 1.0 / (cost_norm + 0.1)  # cheaper = higher efficiency
    
    # Final score: weighted combination
    # Higher weight on EI for exploitation, UCB for exploration
    score = (ei_norm * 0.6 + ucb_norm * 0.4) * cost_efficiency
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
