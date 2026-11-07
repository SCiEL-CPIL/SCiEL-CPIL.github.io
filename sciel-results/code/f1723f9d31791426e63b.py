"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a single sample from the GP posterior and scores each candidate
    by the negative of the sampled value divided by its cost. This naturally
    balances exploration and exploitation without tuning a beta parameter.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each point (1 + mean(x) for x in [0,1]^6).
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the posterior: mu + sigma * N(0,1)
    noise = np.random.default_rng().standard_normal(len(mu))
    sampled_values = mu + sigma_safe * noise
    
    # Score: lower sampled value is better, penalize by cost
    # Use negative sampled value divided by cost (higher score = better)
    # To avoid division issues, use a soft penalty
    scores = -sampled_values / (cost ** 0.5)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
