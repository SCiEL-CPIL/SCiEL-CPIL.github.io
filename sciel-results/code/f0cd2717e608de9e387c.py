import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Expected Improvement (EI) for minimization
    # This balances exploration and exploitation more effectively than fixed UCB
    # for multimodal functions like Branin.
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Calculate Z-score
    Z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Clip to ensure non-negative scores (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    return ei
