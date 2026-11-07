"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Use a hybrid approach: primarily EI, but with a small exploration bonus
    # based on sigma to help escape local optima in early stages.
    # The sphere function is convex, so we want to balance exploitation and exploration.
    
    # Floor sigma to avoid division by zero and extreme values
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Add a small exploration term: proportional to sigma
    # This helps when EI is very small (i.e., mu is close to f_best) but uncertainty is high
    # We scale it by a small factor to not dominate EI
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
