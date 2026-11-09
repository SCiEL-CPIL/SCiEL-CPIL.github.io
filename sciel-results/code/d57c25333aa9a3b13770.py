"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (CA-EI)
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Z score
    Z = (f_best - mu) / sigma_safe
    
    # Standard Normal PDF and CDF
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    # Expected Improvement
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical noise might make it slightly negative)
    EI = np.maximum(EI, 0)
    
    # Cost-awareness: Divide by cost
    # Higher EI and lower cost is better
    scores = EI / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
