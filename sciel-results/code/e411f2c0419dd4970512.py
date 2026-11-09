"""Rewritten program with cost-normalized POI + deterministic regret lower bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Formula: EI(x) / cost(x)
    where EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    and z = (f_best - mu(x)) / sigma(x).
    
    This optimizes the expected reduction in simple regret per unit cost.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-normalized score
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
