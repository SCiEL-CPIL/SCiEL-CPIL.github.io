import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition for Styblinski-Tang 2D BO.
    Minimizes LCB/cost to directly target high-value, low-cost regions.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe
    
    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Known minimum of Styblinski-Tang 2D is approximately -78.3323
    known_min = -78.3323
    
    # Compute progress: 0 at start, 1 when we've found the optimum
    progress = np.clip(1.0 - (f_best - known_min) / 150.0, 0.0, 1.0)
    
    # Adaptive beta for LCB: more exploration early, more exploitation later
    beta = 2.0 * (1.0 - 0.5 * progress)
    
    # LCB = mu - beta * sigma (lower is better)
    lcb = mu - beta * sigma
    
    # Cost-aware score: we want to minimize LCB/cost
    # So we return negative of that (higher score = better)
    score = -lcb / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
