"""Rewritten acquisition using a pure Gaussian Process (GP) posterior mean minimization strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global start time for deadline guard
_global_start_time = None

def acquisition(mu, sigma, f_best):
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Deadline guard: 330s total. We have 300s per eval, but total is 330.
    # If we've been running for > 325s, return a safe fallback immediately.
    # The fallback here is just the negative mean (greedy exploitation).
    if time.time() - _global_start_time > 325.0:
        return -mu

    # Strategy: Greedy exploitation using the GP posterior mean.
    # For minimization, we want to minimize the predicted function value.
    # The score should be higher for points with lower predicted values.
    # So, Score = -mu.
    
    # Why this might be better than EI/UCB:
    # 1. EI and UCB add an exploration term (sigma) or a complex expectation term.
    # 2. In the early stages (first 5-10 points), the GP model is poorly constrained.
    #    High sigma regions are everywhere. EI/UCB might pick points just because they are uncertain, 
    #    potentially missing the true basin of attraction if the initial samples were unlucky.
    # 3. Greedy search (minimizing mu) forces the algorithm to converge on the best region 
    #    identified by the model. If the model is even slightly better than random in the 
    #    "best" region, this will outperform exploration which spreads effort.
    # 4. With only 30 evaluations total, and a smooth function like ST1D, once the model 
    #    identifies a deep basin (e.g., near -5 or +5), it should stay there. 
    #    EI can be "shy" if sigma drops quickly in the basin.
    # 5. This is effectively "greedy hill climbing" guided by the GP mean.
    
    # Ensure finite output
    scores = -mu
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores

# EVOLVE-BLOCK-END
