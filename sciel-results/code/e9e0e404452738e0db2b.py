import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Parameters:
    mu: array of shape (2000,), predicted mean from GP
    sigma: array of shape (2000,), predicted standard deviation from GP
    f_best: float, best function value observed so far
    cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,), acquisition scores (higher is better)
    """
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal PDF and CDF
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * P(X < f_best) + sigma * E[(f_best - X) | X < f_best]
    #     = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: divide by sqrt(cost)
    # This penalizes high cost less than 1/cost, allowing exploration of 
    # slightly more expensive regions which may contain the global minimum.
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / np.sqrt(cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
