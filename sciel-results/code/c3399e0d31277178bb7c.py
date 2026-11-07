"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with a small exploration margin xi to avoid premature convergence.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Small margin to encourage exploration beyond the immediate best
    xi = 0.01
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - xi - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI divided by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
