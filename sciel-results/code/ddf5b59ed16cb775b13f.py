"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid issues, though UCB doesn't divide by it
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB for minimization: maximize -mu + beta * sigma
    # This is equivalent to LCB for minimization: minimize mu - beta * sigma
    # We return the score such that higher is better.
    # Score = -mu + beta * sigma
    
    # Based on inspiration that UCB performed well (score 2.155 mentioned in insights),
    # we use a fixed beta. 1.5 is a standard strong exploration value.
    beta = 1.5
    
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
