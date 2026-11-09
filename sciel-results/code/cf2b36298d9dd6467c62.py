"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling with Cost Awareness
    # Sample from the posterior distribution N(mu, sigma^2)
    # We want to minimize the objective, so we maximize the negative sample.
    # Cost is in [1, 2]. We divide by cost to prefer cheaper evaluations.
    
    # Ensure sigma is positive to avoid division by zero or invalid normal sampling
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Draw a sample from the Gaussian posterior for each candidate
    # np.random.normal takes loc and scale
    samples = np.random.normal(mu, sigma_safe)
    
    # Calculate cost-aware score
    # Higher score is better.
    # We want low function values (minimization) and low cost.
    # Score = -Sample / Cost
    # If Sample is very low (good), -Sample is high (good).
    # If Cost is low, the fraction is larger (good).
    
    # Handle potential edge cases where cost might be 0 (though task says cost >= 1)
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    # If any NaN or Inf occur, replace with a very low score (or 0) to avoid invalidating the result
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
