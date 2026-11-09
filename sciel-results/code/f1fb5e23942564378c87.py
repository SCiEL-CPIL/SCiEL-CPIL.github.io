import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CEI).
    
    Parameters:
    mu: (2000,) array of GP means in objective units
    sigma: (2000,) array of GP stds in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns:
    (2000,) array of scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Calculate z-score for Expected Improvement
    # For minimization, improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If z is very large negative, Phi(z) is 0, phi(z) is 0. EI is 0.
    # If z is very large positive, Phi(z) is 1, phi(z) is 0. EI is f_best - mu.
    
    # Use scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted Expected Improvement
    # We want to maximize improvement per unit cost
    score = ei / cost
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
