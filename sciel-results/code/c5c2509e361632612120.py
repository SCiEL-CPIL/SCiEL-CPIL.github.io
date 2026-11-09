import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # We maximize this score.
    
    # Handle zero or near-zero sigma to avoid division issues
    # If sigma is 0, the point is deterministic.
    # If mu < f_best, EI = f_best - mu.
    # If mu >= f_best, EI = 0.
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI should be non-negative, but numerical errors might cause tiny negatives.
    # Clip to 0 to be safe, though it shouldn't significantly impact ranking.
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
