"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (CEI) acquisition function.
    
    This approach maximizes the expected improvement in the objective function
    per unit of acquisition cost. It is robust for sparse data as it balances
    exploration (via the sigma term in EI) and exploitation (via the mu term).
    
    Formula:
        z = (f_best - mu) / sigma
        EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        Score = EI / cost
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi(z) is CDF, phi(z) is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-effective score
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Replace non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
