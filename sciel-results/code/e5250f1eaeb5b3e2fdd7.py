"""Rewritten program using Thompson Sampling for cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Implements standard Thompson Sampling by sampling a function value
    theta_i ~ N(mu_i, sigma_i^2) for each candidate i, and scoring
    it as -theta_i / cost_i. Higher scores indicate candidates with
    lower predicted function values and lower evaluation costs.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division/sampling issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Sample theta_i from the GP posterior for each candidate
    # theta_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1)
    z = np.random.standard_normal(size=mu.shape)
    theta = mu + sigma * z
    
    # Calculate cost-normalized scores
    # We want to minimize theta (function value) and cost.
    # Score = -theta / cost.
    # If theta is low (good), -theta is high (good).
    # If cost is low (good), division increases score (good).
    scores = -theta / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
