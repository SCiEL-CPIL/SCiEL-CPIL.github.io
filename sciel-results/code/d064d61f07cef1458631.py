"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)"""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement (EI) with cost normalization.
    
    We compute the expected improvement over f_best for each candidate,
    then normalize by the evaluation cost. Higher scores correspond to
    candidates with higher expected improvement per unit cost.
    """
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost safe
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    # Import scipy for CDF and PDF
    from scipy.stats import norm
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.cdf(z)
    phi_pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * phi_z + sigma_safe * phi_pdf
    
    # Ensure EI is non-negative (it should be, but numerical issues might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
