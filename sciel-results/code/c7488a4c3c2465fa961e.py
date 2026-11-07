import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) and phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI can be negative if mu > f_best and sigma is small.
    # We usually clip to 0, but for ranking, negative values are fine (they will be lower than positive).
    # However, if all are negative, the highest (least negative) is chosen.
    
    # Ensure finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
