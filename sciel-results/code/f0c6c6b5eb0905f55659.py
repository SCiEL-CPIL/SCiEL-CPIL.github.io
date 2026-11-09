"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 330.0  # seconds, hard kill
_START_TIME = None
_CALL_COUNT = 0

def _ensure_deadline():
    """Track elapsed time and raise if we're approaching the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 30:  # 30s safety margin
        raise TimeoutError("Approaching deadline")

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    mu: array of shape (N,) - GP mean predictions
    sigma: array of shape (N,) - GP standard deviation predictions
    f_best: float - Best observed value so far
    cost: array of shape (N,) - Cost of evaluating each point
    
    Returns:
    scores: array of shape (N,) - Acquisition scores (higher is better)
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    _ensure_deadline()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Use call count as a proxy for progress
    # Early: beta ~ 3.0, Late: beta ~ 0.5
    # Decay over roughly 50-100 calls (typical budget allows ~25-30 evals)
    t = min(_CALL_COUNT, 100.0)
    beta = 3.0 - 2.5 * (t / 100.0)
    beta = max(beta, 0.5)
    
    # Cost-normalized LCB: (mu - beta * sigma) / cost
    # Higher score = more promising per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB lower bound (we want to minimize, so lower is better)
    lcb = mu - beta * sigma_safe
    
    # We want to maximize the score, so we negate LCB and divide by cost
    # This gives us "improvement density" - how much we expect to improve per unit cost
    scores = -lcb / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
