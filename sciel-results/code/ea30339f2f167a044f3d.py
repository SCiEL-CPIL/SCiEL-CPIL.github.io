"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition for minimization.
    Lower scores in mu - kappa * sigma are better, so we return
    negative of that (higher is better for selection).
    
    We use an adaptive kappa that balances exploration and exploitation.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: larger when we're uncertain, smaller when exploiting
    # Use a fixed kappa that works well for this problem
    kappa = 2.0
    
    # LCB: lower confidence bound (lower is better for minimization)
    lcb = mu - kappa * sigma
    
    # Return negative so higher scores mean "evaluate next"
    scores = -lcb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
