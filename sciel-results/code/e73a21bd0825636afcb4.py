"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) strategy.
    # Unlike EI, which is a non-linear function of the Z-score, UCB is linear in sigma.
    # This promotes exploration of high-uncertainty regions more aggressively,
    # which is beneficial for small budgets (25 points) to avoid local optima.
    
    # Exploration parameter
    beta = 2.0
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # We want to minimize f, so we maximize the score: -mu + beta * sigma
    # Higher score means more promising (either known low value or high potential for low value).
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
