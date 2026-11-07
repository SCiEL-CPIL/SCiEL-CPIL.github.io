"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We subtract the uncertainty term to encourage exploration.
    # kappa controls the balance between exploitation (mu) and exploration (sigma).
    # A value of 1.5 is a common robust choice for 2D problems.
    kappa = 1.5
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite. 
    # If sigma or mu contains NaN/Inf, replace with a very large negative number 
    # so that these points are never selected (since we maximize the score to find the next point).
    # However, the GP should produce finite values. 
    # Just in case, we replace non-finite values with -inf.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
