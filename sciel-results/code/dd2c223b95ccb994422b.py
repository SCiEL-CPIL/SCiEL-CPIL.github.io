"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) with exploration bonus.
    # Combines PI with a small exploration term to balance exploitation and exploration.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima while still favoring high PI regions
    exploration_weight = 0.1
    exploration = -sigma / (np.max(sigma) + 1e-9)
    
    return pi + exploration_weight * exploration
# EVOLVE-BLOCK-END
