"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array of shape (N,), posterior mean
    sigma: array of shape (N,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    start_time = time.time()
    
    # Thompson Sampling: Draw a sample from the posterior
    # z ~ N(0, 1)
    z = np.random.normal(0, 1, size=mu.shape)
    
    # y = mu + sigma * z
    y = mu + sigma * z
    
    # We want to minimize the function, so we maximize the negative sample
    scores = -y
    
    # Deadline guard (though this operation is very fast)
    elapsed = time.time() - start_time
    if elapsed > 250: # Safety margin before 300s limit (though this is per call, 
                      # the limit is per run, but we guard against pathological slowness)
         # If we were doing heavy work, we'd return a fallback. 
         # Here, it's just vector ops, so this is effectively a no-op guard.
         pass
         
    return scores
# EVOLVE-BLOCK-END
