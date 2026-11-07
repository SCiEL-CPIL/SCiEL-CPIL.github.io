import numpy as np
import time

# Global state to track optimization progress and handle potential timeouts
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    
    # Initialize timer on first call
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: 330s limit, keep 30s margin
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return a safe fallback: high score for the candidate closest to the mean of mu?
        # Or just random? The contract says "anything else makes the whole result invalid".
        # Returning zeros is safe.
        return np.zeros_like(mu)

    # Thompson Sampling:
    # Draw a single sample from the GP posterior: f_sample = mu + sigma * z
    # where z is a standard normal random variable.
    # We want to minimize f_sample.
    # The evaluator maximizes the returned score.
    # So we return -f_sample.
    
    # Note: The problem states "mu and sigma ... are in objective units".
    # Standard GP posterior sampling: f(x) ~ N(mu(x), sigma^2(x))
    
    # Generate standard normal noise
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from posterior
    f_sample = mu + sigma * z
    
    # Return negative sample so that the minimum of f_sample becomes the maximum score
    scores = -f_sample
    
    # Safety checks: ensure finite scores
    # If sigma is 0 somewhere, f_sample is deterministic.
    # If mu or sigma are NaN/Inf, handle it.
    if not np.all(np.isfinite(scores)):
        # Replace non-finite values with a very low score so they are never picked
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        # If all are -inf, return zeros to avoid invalid result
        if np.all(scores == -np.inf):
            return np.zeros_like(mu)
            
    return scores
