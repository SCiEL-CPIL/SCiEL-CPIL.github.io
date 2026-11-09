import numpy as np
from scipy.stats import norm
import time

# Track call index to adapt strategy over the optimization loop
_call_count = [0]
_start_time = [time.time()]

def acquisition(mu, sigma, f_best):
    _call_count[0] += 1
    step = _call_count[0]
    
    # Deadline guard
    # The limit is 330s. We return a safe default if we are close to the limit.
    # Since this function is very fast, this is unlikely to trigger.
    if time.time() - _start_time[0] > 325:
        return -np.ones(len(mu))

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Adaptive Beta strategy
    # Total 25 acquisition steps (30 evals - 5 initial)
    # Early steps: High exploration
    # Late steps: High exploitation
    
    if step <= 5:
        beta = 1.5
    elif step <= 10:
        beta = 1.0
    elif step <= 15:
        beta = 0.5
    elif step <= 20:
        beta = 0.2
    else:
        beta = 0.1
        
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
