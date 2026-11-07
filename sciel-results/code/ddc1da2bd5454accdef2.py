import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
    
    # Calculate Z-score for Expected Improvement
    # For minimization, improvement is f_best - f(x)
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu < f_best, z > 0, Phi(z) ~ 1, EI ~ f_best - mu.
    # If mu > f_best, z < 0, Phi(z) ~ 0, EI ~ 0.
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-effective score: EI per unit cost
    # Cost is at least 1, so division is safe.
    score = ei / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
