import numpy as np
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback: return UCB with minimal exploration to be safe
        return -mu + 0.1 * sigma

    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look for low mu.
    # Score = -mu + beta * sigma
    # Higher score is better.
    
    # Decay beta to shift from exploration to exploitation
    # Start with strong exploration, decay over ~25 steps (30 total - 5 initial)
    beta = 5.0 / (1.0 + 0.2 * _call_count)
    
    ucb = -mu + beta * sigma
    
    # Ensure finite scores
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    return ucb
