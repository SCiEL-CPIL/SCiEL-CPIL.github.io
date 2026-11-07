"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimization.
    
    We sample from the posterior: y = mu - sigma * z where z ~ N(0,1)
    (negative because we want lower values to be better).
    The score is the negative of the sampled value, so higher scores
    correspond to lower (better) sampled objective values.
    """
    # Thompson Sampling: sample one value from the posterior at each point
    # For minimization, we want the lowest sampled value
    # y_i = mu_i - sigma_i * z_i where z_i ~ N(0,1)
    # Score = -y_i = -mu_i + sigma_i * z_i
    # Higher score = lower sampled value = better
    
    # Use a fixed seed per call would be bad; use random
    z = np.random.standard_normal(len(mu))
    sampled = mu - sigma * z
    # Return negative so that higher score means lower (better) sampled value
    return -sampled
# EVOLVE-BLOCK-END
