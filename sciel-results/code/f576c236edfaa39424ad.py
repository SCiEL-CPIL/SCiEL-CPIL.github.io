import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Aware Expected Improvement (CEI)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle small sigma to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # If sigma was very small, the formula above is an approximation.
    # More accurately, if sigma -> 0, EI -> max(0, f_best - mu).
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) behaves well as z -> +inf (Phi->1, phi->0) 
    # and z -> -inf (Phi->0, phi->0).
    # Let's explicitly set EI for very small sigma to be safe against numerical noise.
    small_sigma_mask = sigma < eps
    ei[small_sigma_mask] = np.maximum(0, f_best - mu[small_sigma_mask])
    
    # Cost-aware score: maximize EI per unit cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
