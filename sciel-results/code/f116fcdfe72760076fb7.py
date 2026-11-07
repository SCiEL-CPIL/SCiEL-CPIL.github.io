"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time and enforce a safety margin
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    elapsed = _get_elapsed()
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        # Return a simple LCB-based score without heavy computation
        beta = 1.0
        score = -mu + beta * sigma / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    # Cost-scaled LCB: prioritize low-cost, high-uncertainty regions
    # In the short 30-cost budget, this avoids wasting evaluations on expensive
    # far corners and reaches the Rosenbrock valley sooner than pure EI/cost
    
    # Adaptive beta: start with more exploration, reduce as budget depletes
    # cost ranges from ~1 (cheap) to ~2 (expensive)
    # Use a moderate exploration parameter
    beta = 1.5
    
    # LCB: lower is better, so we negate for "higher score = evaluate next"
    # score = -mu + beta * sigma / cost
    # This favors points with low predicted mean (exploitation)
    # and high uncertainty at low cost (exploration)
    score = -mu + beta * sigma / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
