"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior at each candidate.
    # The posterior is N(mu, sigma^2). We return the negative sample so that
    # higher scores correspond to lower predicted objective values (since we minimize).
    # This is a stochastic acquisition: each call to acquisition (if it were called
    # multiple times) would yield different samples, but here we only call it once
    # per step on 2000 candidates, so we draw one sample per candidate.
    
    # Draw samples from N(mu, sigma^2)
    # sigma is already the standard deviation
    samples = mu + sigma * np.random.standard_normal(size=mu.shape)
    
    # We want to minimize, so higher score = better (lower predicted value)
    # Return negative samples so that the argmax picks the lowest predicted value
    return -samples
# EVOLVE-BLOCK-END
