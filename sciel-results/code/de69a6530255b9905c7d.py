import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe default
    if time.time() > _DEADLINE - 5.0:
        # Return a simple LCB score that's still valid
        sigma_safe = np.maximum(sigma, 1e-9)
        # Beta of 2.0 is a standard choice for LCB
        lcb = mu - 2.0 * sigma_safe
        # Cost-aware score: LCB / cost
        # Since we are minimizing, lower mu is better.
        # LCB = mu - beta*sigma. We want to maximize the acquisition score.
        # So we use -LCB / cost? Or (mu - beta*sigma) / cost?
        # Wait, the task says "Return 2000 finite scores (higher = evaluate next)".
        # If we minimize f, we want low f.
        # LCB gives a lower bound on f.
        # If we want to find the minimum, we should pick the point with the lowest LCB.
        # But the interface requires HIGHER scores to be evaluated.
        # So we should return -LCB / cost.
        
        score = -lcb / np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure sigma is non-negative and not zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate LCB
    # beta controls exploration. 2.0 is a common default.
    # Since f_best is provided but LCB doesn't strictly need it (it's absolute),
    # we ignore f_best for the core calculation, or we could use a relative LCB.
    # Standard GP-LCB: min_x mu(x) - beta * sigma(x)
    # To maximize score: max_x - (mu(x) - beta * sigma(x)) / cost(x)
    
    beta = 2.0
    lcb = mu - beta * sigma
    
    # Score is -LCB / cost
    score = -lcb / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
