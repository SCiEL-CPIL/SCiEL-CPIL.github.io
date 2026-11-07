"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition function using Expected Improvement (EI) for minimization.
    # EI measures the expected improvement over the current best (f_best).
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # We divide by cost to get improvement per unit cost.
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical overflow/underflow in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    scores = ei / cost
    
    # Handle any numerical instabilities
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
