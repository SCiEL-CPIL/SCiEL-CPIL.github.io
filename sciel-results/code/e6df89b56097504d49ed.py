"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior predictive
    # at each candidate and return the negative (since lower is better).
    # The sample is mu - sigma * z where z ~ N(0, 1).
    # We want to minimize, so we pick the candidate with the lowest sampled value.
    # Score = -sampled_value, so higher score means lower sampled value.
    rng = np.random.default_rng()
    z = rng.standard_normal(size=mu.shape)
    sampled_values = mu - sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
