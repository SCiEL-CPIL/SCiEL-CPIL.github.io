import numpy as np
import time

# Global start time for the deadline guard
_start_time = time.time()
_DEADLINE = 290.0  # Safety margin before 300s

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback, e.g., random or zero
        return np.zeros_like(mu)
    
    # GP-LCB formulation for minimization
    # We want to maximize (beta * sigma - mu)
    # This is equivalent to minimizing (mu - beta * sigma)
    beta = 2.0
    
    # Cost-aware adjustment
    # Use sqrt(cost) to soften the penalty as per insights
    cost_penalty = np.sqrt(cost)
    
    # Calculate score
    score = (beta * sigma - mu) / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
