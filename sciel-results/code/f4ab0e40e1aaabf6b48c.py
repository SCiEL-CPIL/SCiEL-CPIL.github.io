import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    pdf_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Calculate Expected Improvement (Minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative. Clip to 0 to handle numerical noise.
    ei = np.maximum(ei, 0.0)
    
    # Score: Expected Improvement divided by cost
    # Higher EI means higher expected gain. Dividing by cost makes it cost-aware.
    scores = ei / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
