"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimizing the Beale function.
    
    Draw a single sample from the posterior predictive at all 2000 candidates:
        y ~ N(mu, sigma^2)
    Return scores = -y so that the candidate with the highest score has the
    lowest sampled objective value.
    """
    # Draw standard normal noise
    z = np.random.standard_normal(len(mu))
    
    # Posterior predictive sample
    y = mu + sigma * z
    
    # Score: higher is better for selection (we want lowest y)
    scores = -y
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
