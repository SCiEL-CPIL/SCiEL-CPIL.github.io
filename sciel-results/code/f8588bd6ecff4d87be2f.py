"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0 # 300s limit, keep margin


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Return a safe fallback: just exploit the best mu, penalized by cost
        # This is a structural change: fallback to pure exploitation under time pressure
        return -mu / cost

    # Cost-Adaptive UCB
    # Exploration term is scaled by 1/cost
    # Beta is set to a value that balances exploration and exploitation
    # Using a fixed beta of 2.0 is a common starting point for UCB in BO
    # However, to be robust to scale, we can normalize sigma by its median
    # But simple fixed beta often works if the GP is well-calibrated.
    
    # Let's try a beta that adapts slightly to the uncertainty level
    # If sigma is large, we explore more.
    # beta = 2.0
    
    # Alternative: Use the EI formulation but with a "sharpness" parameter?
    # No, let's stick to the UCB structural change.
    
    # To handle scale invariance, we can normalize the score by the median sigma
    # This makes the exploration term comparable to the exploitation term
    med_sigma = np.median(sigma) + 1e-9
    beta = 2.0
    
    # Score = (-mu + beta * sigma) / cost
    # We want to maximize this.
    # To avoid numerical issues, ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    score = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
