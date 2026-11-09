"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) approach for minimization.
    # EI(x) = E[max(f_best - f(x), 0)]
    # This metric directly estimates the expected reduction in the best observed value.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition. Clip to 0 to handle numerical artifacts.
    scores = np.maximum(scores, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
