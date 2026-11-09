"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    For each candidate point, we draw a single sample from the Gaussian posterior
    N(mu, sigma^2). The candidate with the lowest sampled value is selected for
    evaluation. This approach implicitly balances exploration (high sigma regions
    have a higher chance of producing a low sample) and exploitation (low mu regions
    are more likely to produce low samples).
    
    Parameters:
    -----------
    mu : array-like, shape (N,)
        Mean of the GP posterior for the candidate points.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior for the candidate points.
    f_best : float
        The best observed value so far. (Not used directly in the score, 
        but part of the interface).
        
    Returns:
    --------
    scores : numpy array, shape (N,)
        Scores for each candidate. Higher scores indicate higher priority for evaluation.
        We return the negative of the sampled value, so that the candidate with the
        lowest predicted function value (best for minimization) has the highest score.
    """
    # Ensure inputs are numpy arrays for efficient vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle potential numerical issues where sigma might be 0 or negative
    # Standard deviation must be non-negative. If it's 0, the sample is just mu.
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Draw a standard normal random sample for each candidate
    # This creates a vector of N random numbers ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    
    # Compute the Thompson Sampling estimate: f_sample = mu + sigma * z
    # This represents a plausible value of the function at each candidate point
    f_sample = mu + sigma_safe * z
    
    # The goal is to minimize the function.
    # We want to return scores where "higher is better".
    # Therefore, we negate the sampled values.
    # The candidate with the lowest f_sample (best guess for minimum) will have the highest score.
    scores = -f_sample
    
    return scores
# EVOLVE-BLOCK-END
