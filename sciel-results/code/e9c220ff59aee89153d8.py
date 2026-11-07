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
    
    # Conditional Expected Gain given Improvement (QI-like)
    # G = EI / P(Improvement)
    # To avoid division by zero when P(Improvement) is tiny, add epsilon
    cond_gain = ei / (cdf + 1e-9)
    
    # Use Logarithmic transformation to compress the dynamic range
    # This prevents regions with very high potential gain (but low probability)
    # from completely dominating the search, aiding exploration in multimodal landscapes.
    log_cond_gain = np.log(cond_gain + 1e-9)
    
    # Cost-aware score
    # Higher score is better
    # We divide by cost to account for the expense
    scores = log_cond_gain / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
