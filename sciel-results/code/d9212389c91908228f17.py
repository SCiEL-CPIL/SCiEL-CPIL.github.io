import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin from 330s limit

def acquisition(mu, sigma, f_best):
    # Time guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to pure exploitation (minimize mu)
        return -mu.copy()
        
    # Thompson Sampling with amplified exploration
    # Knowledge indicates that increasing the effective standard deviation
    # (kappa > 1) improves performance on Beale by escaping local optima.
    # Measured improvement with kappa=1.5. Trying kappa=2.0 for further exploration.
    kappa = 2.0
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma * z * kappa
    
    # Return negative samples so higher score = lower predicted value
    return -samples
