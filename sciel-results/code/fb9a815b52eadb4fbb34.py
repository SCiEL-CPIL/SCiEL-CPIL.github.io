"""Rewritten program for Bayesian optimization of ackley 2d with Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draws a single random sample from the posterior at each candidate point:
    theta = mu + sigma * Z, where Z ~ N(0, 1)
    
    Returns negative sample as score (higher score = better for minimization).
    """
    # Generate standard normal noise for all candidates
    Z = np.random.standard_normal(mu.shape)
    
    # Draw posterior sample
    theta = mu + sigma * Z
    
    # Return negative of sample (higher is better for minimization)
    return -theta
# EVOLVE-BLOCK-END
