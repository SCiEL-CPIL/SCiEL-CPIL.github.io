import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start = time.time()
    limit = 1.5  # 1.5 seconds limit per call, safety margin
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats for normal CDF and PDF
    from scipy.stats import norm
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # If it takes too long, fallback to simple greedy (lower mu is better)
    if time.time() - start > limit:
        scores = -mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
