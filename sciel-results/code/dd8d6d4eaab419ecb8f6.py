import numpy as np
import time

_start_time = time.time()
_CALL_COUNT = 0
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    global _CALL_COUNT, _start_time
    
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - 5.0:
        _CALL_COUNT += 1
        return -mu / (cost + 1e-9)

    _CALL_COUNT += 1
    
    # Estimate total cost consumed
    # Average cost per eval is approx 1.5 (1 + 0.5)
    # 5 initial points + N calls
    # Initial cost approx 7.5
    # Current cost approx 7.5 + 1.5 * N
    est_total_cost = 7.5 + 1.5 * _CALL_COUNT
    
    # Adaptive Beta
    # beta = 1 / (1 + total_cost/30)
    # Let's make it stronger:
    # beta = 2.0 / (1.0 + est_total_cost / 30.0)
    
    beta = 2.0 / (1.0 + est_total_cost / 30.0)
    
    # CA-UCB for minimization
    # L(x) = mu - beta * sigma
    # Score = (f_best - L(x)) / cost
    # This rewards points that are likely to be lower than f_best, 
    # weighted by uncertainty, and penalized by cost.
    
    lcb = mu - beta * sigma
    scores = (f_best - lcb) / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
