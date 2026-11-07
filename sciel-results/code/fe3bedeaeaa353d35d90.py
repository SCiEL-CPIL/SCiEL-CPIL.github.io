"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize mu - k*sigma.
    # The interface requires higher scores to mean "evaluate next".
    # So we return - (mu - k*sigma) = -mu + k*sigma.
    
    k = 3.0
    
    # Calculate scores
    scores = -mu + k * sigma
    
    # Ensure all scores are finite.
    # Replace NaNs and Infs with values that will not be selected.
    # Since valid scores are typically positive (as mu is negative for Michalewicz min),
    # a large negative number is a safe "do not select" value.
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
