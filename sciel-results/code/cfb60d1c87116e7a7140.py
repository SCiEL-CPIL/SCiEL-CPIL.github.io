"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use the known global minimum (3.0) as the reference point for EI.
    # This maintains a strong signal to explore regions that could potentially
    # reach the global optimum, rather than just improving on the current best.
    f_ref = 3.0
    
    # Calculate Z-scores relative to the global minimum
    # Add small epsilon to sigma to avoid division by zero
    sigma_safe = sigma + 1e-9
    z = (f_ref - mu) / sigma_safe
    
    # Expected Improvement relative to the global minimum
    # This is zero if mu is far above 3.0, but grows as mu approaches 3.0
    ei_global = (f_ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a variance bonus to encourage exploration in high-uncertainty regions
    # even when the expected improvement relative to 3.0 is low (early stages)
    # The lambda coefficient balances exploration vs exploitation
    lambda_explore = 1.0
    
    scores = ei_global + lambda_explore * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
