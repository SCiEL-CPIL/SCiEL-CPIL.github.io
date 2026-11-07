"""Rewritten program using Thompson Sampling acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    For each candidate, draw a sample from the posterior N(mu, sigma^2)
    and score candidates by their sampled values (lower is better).
    We return negative samples so that higher score = lower predicted value.
    
    Parameters:
    -----------
    mu : array of shape (n_candidates,)
        Posterior mean at each candidate
    sigma : array of shape (n_candidates,)
        Posterior standard deviation at each candidate
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (n_candidates,)
        Acquisition scores; higher means evaluate next
    """
    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Draw Thompson samples: y ~ N(mu, sigma^2)
    # Use a random seed derived from the current state to ensure
    # different samples each call while being reproducible
    # Use the hash of f_best and mu's mean to seed
    seed_val = int(abs(hash((f_best, float(np.mean(mu)), n))) % (2**32))
    rng = np.random.RandomState(seed_val)
    
    eps = rng.standard_normal(n)
    samples = mu + sigma * eps
    
    # Return negative samples: lower sampled value -> higher score
    # Ensure all scores are finite
    scores = -samples
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
