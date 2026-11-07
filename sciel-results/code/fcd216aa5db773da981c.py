"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # Ensure sigma is non-zero to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization: we want to maximize -mu + exploration term
    # Use a hybrid of exploitation and exploration
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Use a combination: primarily EI, but add small UCB component for exploration
    # UCB for minimization: -mu + beta * sigma
    beta = 0.1
    ucb = -mu + beta * sigma
    
    # Normalize both to similar scale
    # Use EI as primary, UCB as secondary
    # Scale EI to be comparable
    ei_scale = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_scale = np.max(ucb) - np.min(ucb) if np.max(ucb) > np.min(ucb) else 1.0
    
    ei_norm = ei / ei_scale
    ucb_norm = (ucb - np.min(ucb)) / ucb_scale
    
    # Combine: 70% EI, 30% UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Check deadline
    if time.time() > deadline:
        # Return best available result
        return scores
    
    return scores
# EVOLVE-BLOCK-END
