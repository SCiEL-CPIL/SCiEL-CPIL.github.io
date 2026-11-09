import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Lower Confidence Bound (LCB).
    
    Strategy:
    1. Use LCB for minimization: score_base = -mu + kappa * sigma.
       This directly targets regions with low predicted value and high uncertainty.
    2. Normalize cost to [0, 1] to make the penalty scale-invariant relative to the candidate set.
    3. Subtract a cost penalty: score = score_base - lambda * cost_norm.
       This discourages evaluating expensive points unless their predicted value/uncertainty is significantly better.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP std predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _check_deadline():
        # Return a simple fallback: prioritize uncertainty over cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return sigma_safe / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize cost to [0, 1] range
    # This ensures the cost penalty is comparable across different cost distributions
    c_min = np.min(cost)
    c_max = np.max(cost)
    c_range = c_max - c_min
    
    if c_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - c_min) / c_range

    # LCB parameters
    # kappa controls exploration (uncertainty bonus)
    # lambda_cost controls the strength of the cost penalty
    # Tuned for small-budget BO on 3HC
    kappa = 1.5
    lambda_cost = 0.5
    
    # Base LCB score for minimization (negated for maximization)
    # We want low mu and high sigma
    lcb_score = -mu + kappa * sigma_safe
    
    # Apply cost penalty
    scores = lcb_score - lambda_cost * cost_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
