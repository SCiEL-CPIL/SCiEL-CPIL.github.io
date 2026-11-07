"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw one sample from the predictive posterior
    # score = -sample, higher score means lower predicted value
    # This naturally balances exploration and exploitation
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-10, None)
    
    # Draw a sample from N(mu, sigma^2)
    samples = np.random.normal(mu, sigma)
    
    # Score is negative of the sample (we want to minimize)
    scores = -samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
