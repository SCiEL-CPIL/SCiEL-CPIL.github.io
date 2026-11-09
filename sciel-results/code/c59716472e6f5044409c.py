import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Args:
        mu: Mean predictions (2000,)
        sigma: Standard deviation predictions (2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate (2000,)
        
    Returns:
        scores: Acquisition scores (2000,), higher is better.
    """
    # Avoid division by zero in z calculation
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    
    cdf_z = norm.cdf(z)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    
    # EI should be non-negative, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # cost is always >= 1 according to problem description (1 + mean(x))
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = ei / cost
        
    # Replace NaN and Inf with finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to reasonable range
    scores = np.clip(scores, 0.0, 1e6)
    
    return scores
