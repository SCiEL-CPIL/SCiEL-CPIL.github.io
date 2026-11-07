import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Adaptive Lower Confidence Bound (LCB) acquisition function
    with uncertainty-weighted cost penalty and exploitation floor.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    # Normalize f_best to estimate progress
    # Ackley 2D max is approx 29.
    # Use a safe upper bound.
    f_scale = 25.0
    
    # Adaptive beta: high when f_best is high (early), low when f_best is low (late)
    # beta = beta_min + (beta_max - beta_min) * (f_best / f_scale)
    beta_min = 0.5
    beta_max = 3.0
    
    # Ensure f_best is positive for the ratio, though it should be.
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # LCB for minimization: mu - beta * sigma
    # We maximize the negative: -mu + beta * sigma
    lcb_score = -mu + beta * sigma
    
    # Cost-aware exploration: scale the uncertainty term by a cost-aware factor
    # Cheaper points get a slightly boosted exploration incentive early on
    cost_safe = np.maximum(cost, 1e-6)
    cost_norm = np.clip(cost_safe / 2.0, 0.0, 1.0)  # cost ranges roughly 1-2
    
    # Cost-aware weight: higher weight for cheaper points, but not too aggressive
    # This encourages exploring cheap regions while still considering expensive ones
    cost_weight = 1.0 + 0.3 * (1.0 - cost_norm)  # ranges from 1.0 to 1.3
    
    # Apply cost-aware weight to the uncertainty component only
    sigma_term = beta * sigma * cost_weight
    mu_term = -mu
    
    # Combined score before cost division
    raw_score = mu_term + sigma_term
    
    # Add a small exploitation floor to ensure low-mu points are still considered
    # This prevents the cost division from completely suppressing promising points
    exploitation_floor = 0.05 * np.exp(-0.5 * mu)  # smooth decay with mu
    
    result = (raw_score + exploitation_floor) / cost_safe
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
