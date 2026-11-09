"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) if mu < f_best else 0
    # where z = (f_best - mu) / sigma
    
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    improvement = f_best - mu
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # EI is 0 if improvement <= 0 (i.e., mu >= f_best)
    ei = np.maximum(ei, 0.0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
