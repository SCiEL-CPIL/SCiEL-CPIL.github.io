"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) for minimization
    # PI(x) = P(f(x) < f_best)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    z = (f_best - mu) / sigma_safe
    
    pi = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(pi), pi, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
