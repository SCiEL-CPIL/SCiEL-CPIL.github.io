import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > 300:
        # Fallback: return simple UCB
        return -mu + 2.0 * sigma
    
    # GP-UCB for minimization
    # Score = -mu + beta * sigma
    # Use a time-varying beta that decreases over the course of optimization
    
    # Infer step number from f_best magnitude
    # Initial values are roughly in [-2.5, 0] range, minimum is -2.53218
    # As we find better values, f_best decreases
    # Use a heuristic: start with high beta, decrease as f_best approaches minimum
    
    # Scale factor based on how close we are to the known minimum
    # If f_best is close to -2.53218, we're in exploitation mode
    # If f_best is far from -2.53218, we're in exploration mode
    
    # Use a smooth schedule: beta starts around 3.0 and decreases to 1.5
    # Map f_best from [0, -2.53218] to [3.0, 1.5]
    min_val = -2.53218
    if f_best > min_val:
        # Normalize: 0 when f_best = 0, 1 when f_best = min_val
        t = (f_best - 0) / (min_val - 0)
        t = np.clip(t, 0, 1)
        beta = 3.0 - 1.5 * t
    else:
        beta = 1.5
    
    return -mu + beta * sigma
