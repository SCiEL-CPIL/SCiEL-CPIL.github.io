import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 250.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    # If we are close to the deadline, we might want to behave differently?
    # But we don't know how many iterations are left.
    # Just ensure this function is fast.
    
    # Cost-aware UCB
    # UCB = -mu + beta * sigma
    # Score = UCB / cost
    
    # We need to handle sigma=0
    sigma_safe = np.maximum(sigma, 1e-6)
    
    beta = 1.5 # Exploration parameter
    
    ucb = -mu + beta * sigma_safe
    scores = ucb / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
