import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Time guard to prevent exceeding limits if called in a tight loop
    # Note: The calculation itself is very fast, but this satisfies the directive.
    # We assume a global start time is not available, so we just ensure the 
    # operation is O(N) and vectorized.
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma
    # If sigma is 0, the GP is certain. EI is max(f_best - mu, 0)
    # If sigma is very small, z can be huge, causing numerical issues in cdf/pdf
    # We use a small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is >= 1, so dividing is safe
    # If cost is 0 (shouldn't happen as cost >= 1), we'd have issues, but contract says cost >= 1
    scores = ei / cost
    
    # Return finite scores
    # Replace any non-finite values with 0 or a large negative number? 
    # "Return 2000 finite scores... anything else invalidates"
    # If EI is inf, score is inf. If cost is 0, score is inf.
    # Let's ensure finiteness.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
