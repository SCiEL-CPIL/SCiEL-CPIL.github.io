import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Calculate Conditional Expected Improvement (CEI)
    # CEI = EI / P(Improvement)
    # This is equivalent to f_best - QI, where QI is the conditional mean.
    # We maximize CEI to find points with high expected improvement given that an improvement occurs.
    
    # Handle cases where P(Improvement) is very small to avoid numerical instability
    # If P(Improvement) is negligible, we set CEI to 0.
    ceis = np.zeros_like(ei)
    valid = cdf > 1e-8
    ceis[valid] = ei[valid] / cdf[valid]
    
    # Cost-aware score
    # Higher score is better
    # We divide by cost to account for the expense
    # Using linear CEI instead of log(CEI) to preserve magnitude differences
    scores = ceis / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
