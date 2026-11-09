import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best):
    if time.time() - _start_time > TIME_LIMIT:
        return -mu.copy()

    # Adaptive Beta
    # If f_best is high, we are likely far from the optimum, so explore more.
    # If f_best is low, we are close, so exploit more.
    # Beale min is 0.
    # Typical initial f_best might be > 10.
    # Good f_best < 1.
    
    # Scale beta based on f_best
    # beta = 1.5 + 1.0 * (f_best / (f_best + 1.0))
    # If f_best -> inf, beta -> 2.5
    # If f_best -> 0, beta -> 1.5
    
    # Let's try a simpler adaptive scheme
    # beta = 1.5
    # if f_best > 5.0:
    #     beta = 2.0
    # else:
    #     beta = 1.0
    
    # Smooth version:
    # beta = 1.0 + 1.0 * (1.0 - np.exp(-f_best / 5.0))
    # f_best=0 -> beta=1.0
    # f_best=5 -> beta=1.0 + 1.0*(1-0.37) = 1.63
    # f_best=inf -> beta=2.0
    
    beta = 1.0 + 1.0 * (1.0 - np.exp(-f_best / 5.0))
    
    scores = -mu + beta * sigma
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
