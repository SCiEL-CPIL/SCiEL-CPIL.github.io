"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Pure Lower Confidence Bound (LCB) acquisition with fixed high exploration constant.
    
    For minimization, LCB = mu - k * sigma. We return the negative LCB so that
    higher scores correspond to lower LCB values (i.e., more promising for minimization).
    
    Using k=3.0 to aggressively target high-uncertainty regions, which helps escape
    local minima in the Michalewicz function.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB = mu - k * sigma (lower is better for minimization)
    k = 3.0
    lcb = mu - k * sigma_safe
    
    # Return negative LCB so higher score = better (lower LCB)
    score = -lcb
    
    return score
# EVOLVE-BLOCK-END
