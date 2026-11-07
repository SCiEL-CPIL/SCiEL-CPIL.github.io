import numpy as np
import time

_start_time = None
_DEADLINE = 325.0 # 5 seconds margin from 330s

def acquisition(mu, sigma, f_best):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard
    if elapsed > _DEADLINE:
        # Return a simple, fast score. 
        # If we are this close to the limit, we just want to finish.
        # Pure exploitation is a safe fallback.
        return -mu
    
    # Thompson Sampling Acquisition Function
    # This is a structural change from deterministic UCB/EI to stochastic sampling.
    # We sample a function value from the posterior GP for each candidate.
    # f(x) ~ N(mu(x), sigma(x)^2)
    # Since we are minimizing, the candidate with the lowest sampled value is the most promising.
    # We return the negative of the sample so that higher scores correspond to lower (better) values.
    
    # Ensure sigma is non-negative (it should be, but just in case)
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Generate random normal noise
    # Using np.random.randn is fast and sufficient.
    noise = np.random.randn(len(mu))
    
    # Sample from the posterior
    samples = mu + sigma_safe * noise
    
    # Score: higher is better. Since we minimize f, lower sample is better.
    scores = -samples
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
