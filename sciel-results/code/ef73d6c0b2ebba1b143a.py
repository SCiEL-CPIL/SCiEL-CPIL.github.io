import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Noisy Expected Improvement for minimization
    # EI(x) = (f_best - xi - mu) * Phi((f_best - xi - mu) / sigma) + sigma * phi((f_best - xi - mu) / sigma)
    # Using xi = 0.05 to encourage exploration
    
    xi = 0.05
    
    # Handle division by zero in sigma
    safe_sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    
    z = (f_best - xi - mu) / safe_sigma
    
    # Calculate EI
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - xi - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
