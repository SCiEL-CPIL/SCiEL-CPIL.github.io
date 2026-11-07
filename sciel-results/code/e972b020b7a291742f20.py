"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Minimizes the surrogate's lower confidence bound divided by evaluation cost.
    Returns scores where higher is better (i.e., we negate the cost-adjusted LCB).
    """
    # Use a beta parameter that balances exploration and exploitation
    # A moderate beta helps escape local minima in multi-modal functions
    beta = 2.0
    
    # Lower Confidence Bound: lower is better for minimization
    lcb = mu - beta * sigma
    
    # Cost-adjusted score: we want low LCB and low cost
    # Negate so that higher score = better (lower LCB/cost)
    # Add small epsilon to avoid division by zero
    scores = -lcb / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
