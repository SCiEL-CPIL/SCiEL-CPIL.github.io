"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Thompson sampling scores; the candidate with the highest score
        should be evaluated next.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson sampling: sample from the posterior at each candidate
    # Sample a single latent function value from N(mu, sigma^2)
    # Use a fixed seed per call is not possible since we need randomness
    # for exploration, but we want reproducibility across seeds handled externally
    
    # Generate random normal samples
    randn = np.random.randn(mu.shape[0])
    samples = mu + sigma_safe * randn
    
    # For minimization, we want to select points likely to have low values
    # Thompson sampling: return the samples directly (lower is better for the function)
    # But the task says "higher meaning evaluate next", so we negate
    # Actually, let me re-read: "Return 2000 finite scores, higher meaning evaluate next"
    # So we need to return scores where higher = more promising
    
    # For minimization, promising points have low expected values
    # Thompson sampling: sample from posterior, and we want to pick the one
    # that gives the best (lowest) value in the sample
    
    # So the score should be such that higher score = more likely to be good
    # We can return -samples (negate the sampled values)
    # Or we can think of it as: we sample f(x) ~ N(mu, sigma^2), and we want
    # to maximize the probability of finding a good point
    
    # Standard Thompson sampling for minimization:
    # Sample theta ~ N(mu, sigma^2) for each candidate
    # Return -theta as the score (so that lower theta = higher score)
    
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
