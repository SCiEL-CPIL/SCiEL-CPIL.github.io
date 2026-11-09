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
    
    # Use K-sample Thompson Sampling to enhance exploration.
    # Taking the minimum of K samples from the posterior N(mu, sigma^2) 
    # biases the score towards the lower tail of the distribution, 
    # which acts similarly to UCB but derived from the sampling perspective.
    K = 10
    Z = np.random.standard_normal((len(mu), K))
    samples = mu[:, None] + sigma[:, None] * Z
    min_samples = np.min(samples, axis=1)
    
    # We want to minimize the objective, so higher acquisition score = lower sampled value
    # Return negative min samples so that argmax(scores) corresponds to argmin(samples)
    return -min_samples
# EVOLVE-BLOCK-END
