"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Expected improvement acquisition function with exploration bonus.
    # We use a small jitter to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Avoid numerical issues: clip z to reasonable range
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, handle large z values
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploring uncertain regions
    # This helps escape local optima in the acquisition landscape
    exploration_bonus = 0.01 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
