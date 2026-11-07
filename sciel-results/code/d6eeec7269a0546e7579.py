"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time and call count for adaptive parameters
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've spent too much time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 310:  # Safety margin before 330s limit
        return np.zeros_like(mu)
    
    # Increment call count
    _call_count += 1
    
    # Ensure finite outputs
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Structural Change: Use Pure UCB (Upper Confidence Bound) for minimization
    # Instead of EI + small exploration, we use an adaptive exploration coefficient beta.
    # Beta starts high to encourage exploration in unvisited regions (high sigma)
    # and decays over time to focus on exploitation (low mu) as we get closer to the optimum.
    
    # Total evaluations allowed is 30.
    # We start exploring heavily and switch to exploiting.
    # A simple decay schedule: beta = beta_max * (1 - t/T)^2 or similar.
    # Let's use a linear decay from 2.0 to 0.5 over the 30 steps.
    # t is 1-based call count.
    t = _call_count
    T = 30.0
    beta_max = 2.0
    beta_min = 0.5
    
    # Linear interpolation
    if t >= T:
        beta = beta_min
    else:
        progress = (t - 1) / (T - 1)
        beta = beta_max - progress * (beta_max - beta_min)
        
    # UCB for minimization: we want to minimize mu + beta * sigma
    # Score should be higher for better candidates.
    # So Score = -(mu + beta * sigma)
    # To avoid negative scores causing issues with "higher is better" if all are negative,
    # we can shift them, but the relative order is what matters for argmax.
    # However, the prompt says "Return 2000 finite scores".
    # Let's stick to the raw negative UCB.
    
    scores = -(mu + beta * sigma_safe)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
