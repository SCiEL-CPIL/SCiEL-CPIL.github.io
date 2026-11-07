"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: expected improvement per unit cost
    # with a small exploration bonus for low-cost regions
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Use scipy for accuracy
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-aware: divide by cost
    # Add a small exploration term to encourage exploration in low-cost areas
    # The cost is 1 + mean(x scaled to [0,1]), so lower cost is better
    # We want to balance EI with cost efficiency
    scores = ei / (cost + eps)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
