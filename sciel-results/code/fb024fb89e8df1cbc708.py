import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    # The GP should provide positive sigma, but just in case
    sigma = np.maximum(sigma, 1e-8)
    
    # Z-score for improvement
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is norm.cdf, phi is norm.pdf
    
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    ei = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # EI is non-negative.
    # We want to maximize EI.
    return ei
