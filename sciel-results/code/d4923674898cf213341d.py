import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit per evaluation, use a safe margin like 295s
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    # EI = (f_best - mu) * P(Z <= z) + sigma * pdf(Z = z)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Check time (unlikely to trigger, but required by directive)
    if time.time() > deadline - 5.0:
        # Return the scores calculated so far. Since the calculation is vectorized and instantaneous,
        # this is just a safeguard. If we were doing iterative optimization, we'd break here.
        pass
        
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
