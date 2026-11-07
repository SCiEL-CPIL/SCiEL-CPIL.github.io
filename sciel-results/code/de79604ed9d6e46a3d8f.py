import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (cEI) acquisition function.
    
    Args:
        mu: (N,) array of GP means.
        sigma: (N,) array of GP standard deviations.
        f_best: float, the best observed value so far (lower is better).
        cost: (N,) array of costs for each candidate.
        
    Returns:
        (N,) array of scores. Higher scores are better.
    """
    # Ensure sigma is at least a small epsilon to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: This formula works for minimization where improvement is f_best - f(x)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # For points where sigma was effectively 0, EI should be max(0, f_best - mu)
    # The formula above approximates this well if sigma is small, 
    # but let's explicitly handle the case where sigma < eps to be precise.
    # If sigma < eps, z is huge if f_best > mu, so Phi(z) ~ 1, phi(z) ~ 0.
    # If sigma < eps, z is huge negative if f_best < mu, so Phi(z) ~ 0, phi(z) ~ 0.
    # So EI ~ max(0, f_best - mu). The formula is stable.
    
    # Cost-aware score
    # We want to maximize EI per unit cost
    score = EI / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
