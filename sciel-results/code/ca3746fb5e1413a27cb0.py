"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    This method samples a value from the GP posterior for each candidate point.
    For minimization, we select the point with the lowest sampled value.
    Since the evaluator selects the point with the highest score, we return
    the negative of the sampled value.
    """
    # Ensure inputs are numpy arrays and 1D
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Numerical stability: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw random numbers from standard normal distribution
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    samples = mu + sigma * z
    
    # We want to minimize the function value, so higher score is better for lower samples.
    # Return negative samples to convert minimization to maximization.
    scores = -samples
    
    return scores
# EVOLVE-BLOCK-END
