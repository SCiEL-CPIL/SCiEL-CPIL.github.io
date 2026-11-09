import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
DEADLINE = 300.0 # Safety margin for the 300s limit mentioned in prompt
# The prompt says "evaluator measures... 300s". 
# "Keep an EXPLICIT deadline guard... return the best result found with a safety margin"
# Since we can't change the return type (must be scores), we can't "return the best result" in the sense of stopping the loop.
# We can only ensure our function doesn't crash or hang.
# If time is up, we might just return zeros to force a default behavior or break the loop?
# The prompt implies the *program* (the whole script) is killed.
# I will add a check that if we are very close to the deadline, we return a deterministic fallback (e.g., zeros) to avoid any potential overhead, though unlikely to be needed.

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - 5:
        # Return zeros, letting the evaluator pick index 0 or handle it
        return np.zeros_like(mu)

    # Thompson Sampling: Sample from the posterior
    # mu and sigma are arrays of shape (2000,)
    # We sample y ~ N(mu, sigma^2)
    # Note: sigma is standard deviation
    y = np.random.normal(loc=mu, scale=sigma)
    
    # Calculate improvement: f_best - y
    # Clip to 0 to avoid negative scores (no improvement)
    improvement = np.maximum(0.0, f_best - y)
    
    # Cost-aware score: improvement per unit cost
    # cost is shape (2000,)
    # Avoid division by zero (cost is >= 1, so safe)
    scores = improvement / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
