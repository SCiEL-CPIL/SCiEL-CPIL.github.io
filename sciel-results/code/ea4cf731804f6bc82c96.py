"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 320.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    cost_safe = np.maximum(cost, 1e-6)
    
    # Lower Confidence Bound: mu - beta * sigma
    # We want to minimize, so lower LCB is better.
    # To maximize score (higher = evaluate next), we negate the LCB.
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Adaptive beta: start with moderate exploration, 
    # The LCB directly balances exploitation (mu) vs exploration (sigma)
    # A fixed beta of ~1.5-2.5 is typical for BO. 
    # Let's try beta = 2.0 for balanced exploration/exploitation
    beta = 2.0
    
    # LCB score: higher is better
    # LCB = mu - beta * sigma (we want to minimize this)
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t_start > _DEADLINE:
        return np.zeros_like(scores)
    
    return scores
# EVOLVE-BLOCK-END
