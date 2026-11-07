# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    For minimization, we calculate the Expected Improvement:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    We then normalize by the evaluation cost to get a cost-aware score:
    Score(x) = EI(x) / cost(x)
    
    This encourages points with high expected improvement relative to their cost.
    """
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive for division
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If sigma is very small, z can be large. 
    # We clip z to avoid numerical overflow in exp for phi, though norm.pdf handles it reasonably.
    # However, for very large positive z, Phi(z) ~ 1 and phi(z) ~ 0.
    # For very large negative z, Phi(z) ~ 0 and phi(z) ~ 0.
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, (f_best - mu) is negative, but Phi(z) will be small.
    # The term (f_best - mu) * Phi(z) + sigma * phi(z) is always non-negative.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Calculate cost-aware score
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
