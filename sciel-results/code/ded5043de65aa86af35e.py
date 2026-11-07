"""Rewritten acquisition with cost-aware LCB + time guard."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Lower Confidence Bound per unit cost.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / np.maximum(cost, 1e-9)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Cost-Aware LCB
    # We want to minimize f(x).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the potential for improvement, so we maximize -LCB(x) / cost(x)
    # Score = (beta * sigma(x) - mu(x)) / cost(x)
    
    # Dynamic beta based on the current best value and uncertainty?
    # A fixed beta is a structural change from EI.
    # Let's use beta = 2.0 which is a common exploration constant.
    beta = 2.0
    
    lcb_score = (beta * s - mu) / c
    
    # To ensure we don't pick points with very high mu (bad predictions) even if sigma is high,
    # we can clip or add a penalty. However, LCB naturally handles this if beta is not too large.
    # If mu is very high (bad), the score will be negative.
    # We should probably ensure scores are comparable.
    
    # Add a small EI component to anchor to the current best? 
    # The directive says "different design or modelling family". Pure LCB is distinct.
    # Let's stick to pure LCB but ensure numerical stability.
    
    return lcb_score
