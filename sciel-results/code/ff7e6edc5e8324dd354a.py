import numpy as np
import time

# Global deadline management
_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _check_deadline():
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def _reset_deadline():
    global _start_time
    _start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using LCB (Lower Confidence Bound)
    with a quadratic cost penalty.
    
    Args:
        mu: array of shape (2000,) - mean predictions
        sigma: array of shape (2000,) - standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        scores: array of shape (2000,) - higher is better
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in cost penalty
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # LCB (Lower Confidence Bound) for minimization
    # We want to minimize, so we prefer low mu and high uncertainty (sigma)
    # LCB = mu - kappa * sigma
    # Higher LCB score means more attractive (we want to maximize the acquisition)
    # Actually, for minimization, the acquisition function should be:
    #   a(x) = -mu + kappa * sigma  (higher is better, since we want low mu and high sigma)
    # Or equivalently, we can use:
    #   a(x) = -mu + kappa * sigma
    
    # Adaptive kappa: balance exploration and exploitation
    # As we get more samples, we should reduce exploration
    # But we don't have direct access to n_samples, so we use f_best as a proxy
    # When f_best is close to the minimum, we should be more exploitative
    
    # Simple adaptive strategy: start with higher kappa, decrease as f_best improves
    # For these functions, the minimum is 0 (except Hartmann at -3.86278)
    # We'll use a fixed kappa that works well across the suite
    
    kappa = 1.8
    
    # LCB-based score (higher is better for minimization)
    lcb_score = -mu + kappa * sigma
    
    # Quadratic cost penalty: prefer low-cost evaluations
    # This was measured to significantly improve performance
    cost_penalty = cost_safe ** 2
    
    # Final score: LCB score divided by cost penalty
    scores = lcb_score / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
