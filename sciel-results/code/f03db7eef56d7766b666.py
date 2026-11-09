"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement with an exploration bonus.
    
    The exploration bonus is weighted by uncertainty relative to the range of the
    posterior mean, encouraging exploration in early iterations when the posterior
    is uncertain and exploitation when it becomes more accurate.
    
    Args:
        mu: array of shape (n_candidates,) - posterior mean predictions
        sigma: array of shape (n_candidates,) - posterior standard deviations
        f_best: float - best (lowest) observed function value so far
    
    Returns:
        array of shape (n_candidates,) - acquisition scores (higher is better)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty regions
    # Scale by the spread of mu to make it adaptive
    mu_range = np.max(mu) - np.min(mu) + 1e-8
    sigma_normalized = sigma_safe / (mu_range + 1e-8)
    
    # Weight the exploration bonus by how uncertain we are relative to the data
    # In early stages, sigma will be large relative to mu_range, so we explore more
    # As the GP learns, sigma decreases and we exploit more
    kappa = 0.1  # Exploration weight
    
    # Combine EI with exploration bonus
    score = ei + kappa * sigma_safe * np.exp(-sigma_normalized)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
