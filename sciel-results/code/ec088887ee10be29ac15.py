"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Cost-adjusted LCB: -mu + beta * sigma / cost
    # This penalizes high-cost regions in the exploration term
    # while keeping the exploitation term (mu) invariant.
    
    # Adaptive beta based on total cost budget usage
    # Early: more exploration, Late: more exploitation
    total_cost = cost.sum()
    budget = 30.0
    
    # Beta decreases as we approach the budget limit
    # This encourages more exploitation late in the optimization
    beta = 2.0 * (1.0 - total_cost / budget) + 0.5
    
    # Ensure beta is positive
    beta = max(beta, 0.1)
    
    # Cost-adjusted LCB: higher score = more promising
    score = -mu + beta * sigma / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
