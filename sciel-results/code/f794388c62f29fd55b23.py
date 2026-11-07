import numpy as np
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Parameters for the acquisition function
    # Beta controls the exploration-exploitation tradeoff in LCB
    # Lambda controls the weight of the cost penalty
    beta = 1.5
    lambda_cost = 1.0
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize f, so we look for low mu and high sigma (uncertainty)
    lcb = mu - beta * sigma
    
    # Cost penalty: we want to minimize cost as well
    # cost is passed as an array of shape (2000,)
    penalty = lambda_cost * cost
    
    # Acquisition score: higher is better
    # We combine the objective lower bound and the cost penalty
    score = lcb - penalty
    
    # Ensure numerical stability
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
