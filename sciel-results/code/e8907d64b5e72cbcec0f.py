"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    We maximize -mu + beta * sigma.
    Beta is set to 1.0 for balanced exploration/exploitation.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Basic safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # UCB score: we want to minimize f, so we look for low mu.
    # We add uncertainty sigma to encourage exploration.
    # Score = -mu + beta * sigma
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
