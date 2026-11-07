import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Args:
        mu: Array of GP mean predictions, shape (N,)
        sigma: Array of GP standard deviations, shape (N,)
        f_best: Scalar, best observed value so far.
        cost: Array of evaluation costs, shape (N,)
        
    Returns:
        Array of acquisition scores, shape (N,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu is low (good), f_best - mu is positive.
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    # Use scipy.stats.norm for stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Numerical safety: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    scores = ei / cost
    
    # Handle any remaining non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
