"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and finite to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite; replace non-finite with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in cdf/pdf
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite; replace non-finite with 0.0
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
