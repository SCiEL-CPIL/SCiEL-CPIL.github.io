"""Rewritten program using Cost-Adjusted Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement (CEI)
    # For minimization:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Score = EI / cost
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Compute CDF and PDF
    # norm.cdf and norm.pdf from scipy.stats
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Compute Expected Improvement
    # Note: If mu < f_best, (f_best - mu) is positive, z is positive.
    # EI represents the expected amount by which f(x) improves upon f_best.
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost adjustment
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
