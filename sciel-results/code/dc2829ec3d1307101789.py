"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs to prevent numerical issues
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Calculate standardized improvement z
    z = (f_best - mu) / sigma
    
    # Clip z to a reasonable range to ensure numerical stability of cdf/pdf
    # cdf(-10) is ~ 7e-25, cdf(10) is ~ 1.0
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement (EI)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI is theoretically non-negative, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
