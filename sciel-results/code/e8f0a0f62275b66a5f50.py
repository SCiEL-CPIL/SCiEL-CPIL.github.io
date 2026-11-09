import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Adjusted Lower Confidence Bound (LCB) with adaptive beta.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values for safety
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive beta for LCB
    # f_best starts high (e.g., 20-40) and decreases to 0.
    # We map f_best to a progress value [0, 1] where 1 is "done".
    # Assume max possible value is around 50 for normalization.
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta starts at 3.0 (high exploration) and decays to 1.0 (exploitation)
    beta = 1.0 + 2.0 * (1.0 - progress)
    
    # Lower Confidence Bound for minimization: mu - beta * sigma
    # We want to minimize this, so we maximize -LCB = beta * sigma - mu
    # Normalize by cost
    score = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
