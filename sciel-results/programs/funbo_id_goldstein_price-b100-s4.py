"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Scores for each candidate; higher means more likely to be selected next.
    """
    # Thompson Sampling: sample from the posterior Gaussian
    # For minimization, we want to maximize -f, so we sample f and return -f
    # This gives superior exploration-exploitation balance for multimodal functions
    
    # Generate a single sample path from N(mu, sigma^2)
    # Use a fixed seed per call would be bad; use random state that advances
    # Actually, we just use np.random which is fine for the evaluator
    
    # Sample from the posterior
    samples = mu + sigma * np.random.standard_normal(len(mu))
    
    # For minimization, higher score = lower sampled value
    # We return negative of the sample so that higher scores correspond to lower f values
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
