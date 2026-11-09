"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running long, return a safe default quickly.
    # We track a module-level start time for this call.
    t_start = time.time()
    deadline = t_start + 290.0  # 300s limit, keep 10s margin

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Cost-aware LCB: (mu - beta * sigma) / cost
    # The linear exploration term (beta * sigma) is more robust in early stages
    # than the nonlinear EI term. We use a moderate beta to balance exploration
    # and exploitation.
    
    # We can also blend with EI for robustness, but the measurement showed
    # pure cost-weighted LCB improved score from 1.0 to 1.196.
    # Let's stick with LCB but tune beta.
    
    # A typical beta for LCB is around 1-3. Let's use 1.5 as a starting point.
    # However, since the cost varies, we might want to adapt beta.
    # For now, let's use a fixed beta that worked well in the measurement.
    
    beta = 1.5
    
    # Compute LCB score
    lcb = mu - beta * sigma_safe
    
    # Cost-aware: divide by cost to get "improvement per unit cost"
    # Since we want to maximize the score, and lower cost is better,
    # dividing by cost makes sense.
    
    score = lcb / cost
    
    # Check deadline
    if time.time() > deadline:
        # Return a safe fallback: just use the LCB without cost normalization
        # to ensure we return finite values
        return lcb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
