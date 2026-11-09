import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses a simple, robust Expected Improvement (EI) acquisition function.
    EI is well-suited for exploitation in the later stages of optimization.
    For the sphere function, which is unimodal, EI should converge quickly
    to the optimum once the GP has a reasonable posterior.
    
    We handle numerical stability by clipping sigma and Z values.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score for EI
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues
    Z_clipped = np.clip(Z, -10, 10)
    
    # Compute EI
    phi_Z = norm.pdf(Z_clipped)
    Phi_Z = norm.cdf(Z_clipped)
    
    EI = (f_best - mu) * Phi_Z + sigma * phi_Z
    
    # Handle numerical artifacts
    EI = np.nan_to_num(EI, nan=0.0, posinf=1e6, neginf=0.0)
    
    return EI
