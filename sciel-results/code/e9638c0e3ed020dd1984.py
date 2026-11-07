"""Rewritten cost-aware acquisition: Cost-Effective Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Return 2000 finite scores (higher = evaluate next).
    # Ensure sigma is positive for division
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x). We expect f(x) ~ N(mu, sigma).
    # EI = E[max(0, f_best - f(x))]
    
    # Z-score
    z = (f_best - mu) / sigma
    
    # Standard Normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware Acquisition: Cost-Effective EI
    # We prefer high EI per unit cost.
    score = ei / cost
    
    # Safety: keep everything finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    
    return score
# EVOLVE-BLOCK-END
