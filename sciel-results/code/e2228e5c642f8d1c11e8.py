"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Sample from the posterior distribution N(mu, sigma^2)
    # For minimization, we want to select the point with the lowest sampled value.
    # The acquisition score should be such that the argmax of the score corresponds
    # to the argmin of the sample.
    # sample = mu - sigma * z, where z ~ N(0, 1)
    # To maximize the score to find the min sample, we return -sample.
    # score = -(mu - sigma * z) = -mu + sigma * z
    
    # Generate random normal variables for each candidate
    # Using a fixed seed or random state is not required, but we can use
    # the default global random state.
    z = np.random.randn(len(mu))
    
    # Calculate the score
    score = -mu + sigma * z
    
    return score
# EVOLVE-BLOCK-END
