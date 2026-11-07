"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 310.0  # 330s limit, leave 20s margin


def _robust_norm(arr):
    """
    Robust normalization using median and MAD (Median Absolute Deviation).
    This is more robust to outliers than mean/std normalization, which is
    critical in 8D where a few extreme candidates can skew the scaling.
    """
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad < 1e-10:
        # If MAD is too small, fall back to mean/std or just subtract median
        std = np.std(arr)
        if std < 1e-10:
            return arr - median
        return (arr - np.mean(arr)) / std
    # 0.6745 is the 0.75 quantile of the standard normal distribution
    # scaling MAD to be consistent with std for a normal distribution
    scale = 1.4826 * mad
    return (arr - median) / scale


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the limit, return a quick fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple UCB-like score
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        fallback = (f_best - mu - 1.0 * sigma_safe) / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Safe cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware EI: EI per unit cost
    # Using linear cost penalty as in the high-performing inspiration
    ei_cost = ei / cost_safe
    
    # Exploration component: UCB-like term (Lower Confidence Bound for minimization)
    # We want to minimize, so we look for low mu and high sigma.
    # UCB for minimization is mu - beta * sigma.
    # But since we want "higher is better" scores, we can use -mu + beta * sigma
    # Or simply use the negative of the LCB.
    # Let's stick to the structure of inspiration 1 which worked well:
    # ucb = (f_best - mu - 1.0 * sigma_safe) / cost_safe
    # This is essentially a normalized LCB.
    ucb = (f_best - mu - 1.0 * sigma_safe) / cost_safe
    
    # Normalize both components using robust normalization (median/MAD)
    # This is the structural change from the cell findings
    ei_cost_norm = _robust_norm(ei_cost)
    ucb_norm = _robust_norm(ucb)
    
    # Combine with a 75% exploitation / 25% exploration weight
    # This balance was found effective in the high-performing inspiration
    scores = 0.75 * ei_cost_norm + 0.25 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
