import numpy as np
import time

_start_time = time.time()
_DEADLINE_SECONDS = 325.0

def acquisition(mu, sigma, f_best):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta for UCB
    # gap indicates how much improvement is still possible
    # If gap is large, explore more
    # If gap is small, exploit more
    min_mu = np.min(mu)
    gap = f_best - min_mu
    
    # Normalize gap by typical uncertainty
    sigma_med = np.median(sigma_safe)
    if sigma_med < 1e-8:
        beta = 1.0
    else:
        # Scale factor: if gap is 10x sigma_med, beta is 1 + 2*10 = 21? Too high.
        # Let's use a softer scaling.
        # beta = 1.0 + 2.0 * np.tanh(gap / sigma_med)
        # tanh keeps it bounded [0, 1]. So beta in [1.0, 3.0].
        beta = 1.0 + 2.0 * np.tanh(gap / sigma_med)
        
    score = -(mu + beta * sigma_safe)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
