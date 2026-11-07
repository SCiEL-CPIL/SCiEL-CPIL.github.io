import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT:
        # Return a simple fallback to avoid timeout
        return np.zeros_like(mu)
    
    # UCB acquisition for minimization:
    # We want to find low values. UCB for minimization is mu - beta * sigma
    # (exploit low mu, explore high uncertainty)
    # But we also need cost-awareness.
    
    # Standardize sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB score for minimization: lower is better
    # ucb = mu - beta * sigma
    beta = 2.0
    ucb = mu - beta * sigma_safe
    
    # Cost-aware: we want to prefer lower cost
    # cost is in [1, 2] (1 + mean(x) where x in [0,1])
    # We divide by cost to penalize expensive points
    safe_cost = np.maximum(cost, 1e-9)
    
    # Combine: we want to minimize (ucb * cost) or minimize ucb/cost?
    # Since ucb can be negative (if mu < beta*sigma), dividing by cost
    # amplifies negative ucb values for low cost (good) and penalizes for high cost
    # Actually, let's think about this differently.
    
    # We want a score where higher = more likely to evaluate next
    # Lower ucb is better (closer to minimum)
    # Lower cost is better
    
    # One approach: score = -ucb / cost
    # This rewards low ucb (good prediction) and low cost (cheap)
    
    score = -ucb / safe_cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
