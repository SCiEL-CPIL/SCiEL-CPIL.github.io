"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function for minimization.
    
    Parameters:
    mu (np.array): GP posterior mean, shape (2000,)
    sigma (np.array): GP posterior std, shape (2000,)
    f_best (float): Best observed value so far
    cost (np.array): Evaluation cost per candidate, shape (2000,)
    
    Returns:
    np.array: Acquisition scores, shape (2000,), higher = evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -50, 50)
    
    cdf_val = norm.cdf(z_clipped)
    pdf_val = norm.pdf(z_clipped)
    
    ei = improvement * cdf_val + sigma_safe * pdf_val
    
    # EI should be non-negative (clip small negative values from numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost_safe
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
