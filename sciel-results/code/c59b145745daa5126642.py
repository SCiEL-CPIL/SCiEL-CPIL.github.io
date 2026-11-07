"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Sample from the posterior at each candidate.
    # For minimization, we want the candidate with the lowest sampled value.
    # The interface requires higher scores to be preferred.
    # So we return the negative of the sampled value.
    
    # Generate standard normal samples for each candidate
    z = np.random.standard_normal(len(mu))
    
    # Sample from N(mu, sigma)
    sampled_values = mu + sigma * z
    
    # Return negative so that the lowest sampled value gets the highest score
    return -sampled_values
# EVOLVE-BLOCK-END
