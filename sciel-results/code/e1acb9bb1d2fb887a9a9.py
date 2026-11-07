"""Thompson Sampling acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: sample from the posterior predictive
    at each candidate and select based on the sampled value (lower is better).
    We return scores where higher means more likely to be evaluated next.
    
    Since the loop picks the candidate with the HIGHEST score, and we want
    to pick the one with the LOWEST sampled value (for minimization),
    we return negative of the sampled value.
    
    To make it a proper stochastic acquisition, we use a single random draw
    from the posterior at each candidate point.
    """
    # Guard against zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # This is a single draw from the posterior predictive
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # We want to maximize the score, but lower sampled values are better.
    # So we return negative samples as scores.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
