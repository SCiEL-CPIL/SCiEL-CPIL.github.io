"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    For each candidate, we draw a single sample from the posterior Gaussian N(mu, sigma^2)
    and return the negative of that sample (since we want to maximize the acquisition score,
    and lower objective values are better).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw one sample from each posterior Gaussian N(mu_i, sigma_i^2)
    # We use the standard normal trick: sample = mu + sigma * Z, where Z ~ N(0, 1)
    Z = np.random.standard_normal(len(mu))
    samples = mu + sigma * Z
    
    # We want to minimize the objective, so higher acquisition score = lower sampled value
    # Return negative samples so that argmax(scores) corresponds to argmin(samples)
    return -samples
# EVOLVE-BLOCK-END
