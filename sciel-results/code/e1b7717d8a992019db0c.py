"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware Expected Improvement scores.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Best observed function value (scalar)
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Cost-aware score
    # Using sqrt(cost) to soften the penalty for expensive evaluations
    # This helps explore regions that might be expensive but potentially optimal
    score = ei / np.sqrt(cost + eps)
    
    # Sanitize output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
