"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the 330s limit, return safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just use -mu / cost (deterministic exploitation)
        safe_cost = np.maximum(cost, 1e-9)
        return -mu / safe_cost

    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Adaptive exploration weight
    # Theoretical minimum for cosine8 8d is -0.8.
    # As f_best approaches -0.8, we want to reduce exploration (beta -> 0).
    # If f_best is much worse than -0.8, we want more exploration.
    # Let's define a progress metric: how close are we to the minimum?
    # progress = (f_best - min_val) / (max_val - min_val) ? 
    # Actually, simpler: if f_best is close to min, reduce beta.
    # min_val = -0.8
    # Let's use a decay based on the gap to the minimum.
    # gap = f_best - (-0.8) = f_best + 0.8
    # If gap is small, beta should be small.
    # If gap is large, beta should be larger.
    # Let's map gap to beta.
    # Assume initial f_best might be around 0.0 or higher.
    # Let's use beta = base_beta * exp(-k * (f_best + 0.8))
    # If f_best = -0.8, exp(0) = 1. This doesn't decay to 0.
    # We want beta -> 0 as f_best -> -0.8.
    # So maybe beta = base_beta * (f_best + 0.8) / (max_gap) ?
    # If f_best = -0.8, beta = 0.
    # If f_best = 0.0, beta = base_beta * 0.8 / 0.8 = base_beta.
    # Let's pick max_gap = 1.0 (since range is [-0.8, 0.8] roughly? No, cosine range is [-1, 1] but min is -0.8? 
    # The task says "min -0.8". So the minimum possible value is -0.8.
    # The maximum value of cosine8 8d on [-1,1]^8?
    # cos(x)^8? Or sum of cos? "cosine8 8d" usually means sum of cosines or product?
    # Standard benchmark: f(x) = sum_{i=1}^d cos(x_i) * exp(-x_i^2)? No.
    # "cosine8" often refers to a specific function. 
    # Let's assume the range of values is roughly [-0.8, 1.0] or similar.
    # Let's just use a simple heuristic:
    # beta = 2.0 * (f_best + 0.8) / 1.0
    # If f_best = -0.8, beta = 0.
    # If f_best = 0.2, beta = 2.0.
    # If f_best = 1.0, beta = 3.6.
    # This seems reasonable.
    
    base_beta = 2.0
    gap = f_best - (-0.8)
    # Clip gap to be non-negative
    gap = max(0.0, gap)
    # Normalize gap by a scale factor, say 1.0
    scale = 1.0
    beta = base_beta * (gap / scale)
    
    # Cap beta to avoid excessive exploration if f_best is very bad
    beta = min(beta, 5.0)
    
    # LCB: mu - beta * sigma
    # We want to minimize, so lower LCB is better.
    # Score: higher is better. So score = -LCB / cost = -(mu - beta * sigma) / cost = (-mu + beta * sigma) / cost
    lcb = mu - beta * sigma_safe
    safe_cost = np.maximum(cost, 1e-9)
    score = -lcb / safe_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
