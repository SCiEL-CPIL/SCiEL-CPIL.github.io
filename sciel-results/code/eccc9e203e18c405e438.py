"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function with numerical stability safeguards.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs to handle potential numerical issues from GP inference
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw one sample from each posterior Gaussian N(mu_i, sigma_i^2)
    Z = np.random.standard_normal(len(mu))
    samples = mu + sigma * Z
    
    # Convert to acquisition scores: higher score = lower objective value
    scores = -samples
    
    # Ensure output is finite as required by the task contract
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)
    
    return scores
# EVOLVE-BLOCK-END
