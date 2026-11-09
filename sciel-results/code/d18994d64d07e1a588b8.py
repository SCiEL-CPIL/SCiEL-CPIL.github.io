import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement (EI)
    # EI(x) = E[max(0, f_best - f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Avoid division by zero in z calculation
    # If sigma is very small, z becomes large.
    # We can compute EI directly using stable formulas or handle small sigma.
    
    # Clip sigma to a small positive value to avoid division by zero issues in z
    # However, if sigma is 0, EI should be max(0, f_best - mu).
    # Let's use a safe sigma.
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z
    z = (f_best - mu) / safe_sigma
    
    # Calculate EI
    # phi is the standard normal PDF, Phi is the CDF
    # Using scipy.stats.norm is robust.
    
    # Standard Normal PDF
    phi_z = norm.pdf(z)
    # Standard Normal CDF
    Phi_z = norm.cdf(z)
    
    # EI term
    ei = (f_best - mu) * Phi_z + safe_sigma * phi_z
    
    # Cost-aware score
    # Score = EI / cost
    # Ensure cost is positive
    safe_cost = np.maximum(cost, eps)
    
    scores = ei / safe_cost
    
    # Handle any potential NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
