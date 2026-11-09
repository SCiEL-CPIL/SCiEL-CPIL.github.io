"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) for minimization.
    
    EI directly estimates the expected reduction in the objective function value.
    We normalize by the cost to favor cheaper evaluations that offer good expected improvement.
    """
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei_values = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost. 
    # Using cost directly in the denominator. 
    # Sometimes cost**0.5 or other powers are used, but 1/cost is standard for cost-aware BO.
    # We add a small epsilon to avoid division by zero, though cost >= 1.
    scores = ei_values / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
