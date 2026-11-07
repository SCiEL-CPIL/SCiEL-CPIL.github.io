# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute the CDF and PDF for speed? No, we need to compute for each point.
# But we can use scipy.stats.norm which is vectorized.
# Let's check if scipy is available. Yes.
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    
    # Input validation and sanitization
    # Ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma (avoid division by zero or negative)
    sigma = np.maximum(sigma, 1e-9)
    
    # Handle edge cases for cost (avoid division by zero or negative)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Bounded exploration bonus: sigma/(sigma + 0.1)
    # This is bounded in [0, 1) and prevents exploration from dominating
    exploration_bonus = sigma / (sigma + 0.1)
    
    # Combine EI and exploration with a small weight
    alpha = 0.05
    combined = ei + alpha * exploration_bonus
    
    # Cost-aware acquisition: combined score per unit cost
    # We want to maximize this score
    score = combined / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280:
        # This should not happen as the operations are fast, but just in case
        return np.zeros_like(mu)
        
    return score

# EVOLVE-BLOCK-END
