import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280  # 20s margin
    
    # Input validation and sanitization
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is strictly positive
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z-scores for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Probability of Improvement (PI) and Expected Improvement (EI)
    # We use EI as it is generally superior for global optimization by considering magnitude
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Phi(z) is CDF, phi(z) is PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Term 1: (f_best - mu) * Phi(z)
    # Note: f_best - mu can be negative if mu > f_best.
    # If mu > f_best, z is negative, cdf_z is small, pdf_z is small.
    # The formula holds.
    term1 = (f_best - mu) * cdf_z
    term2 = sigma * pdf_z
    
    ei = term1 + term2
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Using division by cost encourages finding improvements in low-cost regions
    score = ei / cost
    
    # Check deadline
    if time.time() > deadline:
         # Return current best effort
         pass
         
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
