"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (CA-EI)
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    
    # Avoid division by zero or very small sigma
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z score
    improvement = f_best - mu
    Z = improvement / sigma_safe
    
    # Calculate Expected Improvement
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # Cost-awareness: Divide by cost
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
