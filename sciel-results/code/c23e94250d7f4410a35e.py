import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement normalized by cost.
    
    Parameters:
    mu: (2000,) GP mean
    sigma: (2000,) GP standard deviation
    f_best: float, best observed value
    cost: (2000,) cost of evaluating each candidate
    
    Returns:
    scores: (2000,) acquisition scores, higher is better
    """
    eps = 1e-9
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical issues in PDF/CDF
    z = np.clip(z, -10, 10)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / Cost
    scores = ei / (cost + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
