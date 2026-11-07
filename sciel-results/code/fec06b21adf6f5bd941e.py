"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score to select the next point.
    # LCB(x) = mu(x) - kappa * sigma(x)
    # Score(x) = -LCB(x) = -mu(x) + kappa * sigma(x)
    
    # Kappa controls the exploration-exploitation trade-off.
    # Increased kappa to encourage more exploration in early stages
    # to escape local minima in the Ackley function.
    kappa = 1.5
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 0.0)
    
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
