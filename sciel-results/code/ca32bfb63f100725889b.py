import numpy as np
from scipy.stats import norm
import time

# Module-level state to track call count for adaptive exploration
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _call_count, _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Increment call count
    _call_count += 1
    
    # Deadline guard: return a safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        # Return a simple deterministic score
        sigma_safe = np.maximum(sigma, 1e-8)
        return (f_best - mu) / sigma_safe
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive cost exponent: start with softer penalty, increase over time
    # Early on, we want more exploration (lower cost penalty)
    # Later, we want to be more cost-efficient (higher cost penalty)
    # Use a smooth transition from 0.5 to 1.0 over the first 20 calls
    t = min(_call_count / 20.0, 1.0)
    cost_exponent = 0.5 + 0.5 * t  # ranges from 0.5 to 1.0
    
    # Cost-aware score
    score = ei / (cost ** cost_exponent)
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima in the acquisition landscape
    exploration_bonus = 0.1 * sigma / (cost ** cost_exponent)
    
    return score + exploration_bonus
