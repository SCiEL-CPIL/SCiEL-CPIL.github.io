"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Sample from the posterior at each candidate point
    # Since we only have marginal distributions, we sample independently
    # f_sample = mu + sigma * z
    z = np.random.normal(0, 1, size=mu.shape)
    samples = mu + sigma * z
    # We want to minimize f, so we return negative samples to maximize the score
    return -samples
# EVOLVE-BLOCK-END
