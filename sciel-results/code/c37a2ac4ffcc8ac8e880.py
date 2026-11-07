"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates Cost-Aware Expected Improvement (EI) for each candidate.
    
    The acquisition function is defined as EI(x) / cost(x).
    This balances the expected improvement in the objective function against
    the cost of evaluating the point.
    
    Parameters:
    mu: array of shape (2000,), predicted mean of the GP.
    sigma: array of shape (2000,), predicted standard deviation of the GP.
    f_best: float, the best observed value so far.
    cost: array of shape (2000,), the cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (2000,), acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the improvement term
    # Improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    
    # Calculate z-score
    z = diff / sigma_safe
    
    # Expected Improvement (EI)
    # EI = diff * Phi(z) + sigma * phi(z)
    # We use scipy's norm.cdf and norm.pdf for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost-aware adjustment: Divide by cost
    # cost is given as an array of shape (2000,)
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
