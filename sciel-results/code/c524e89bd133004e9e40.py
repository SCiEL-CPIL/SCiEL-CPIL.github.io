import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: z = (f_best - mu) / sigma
    # We want to find points where f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of standard normal
    # P(f(x) < f_best) = P(Z < z) = Phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Quantile of Improvement (QI):
    # We want to maximize the expected value of f_best - f(x) given that f(x) < f_best
    # E[f_best - f(x) | f(x) < f_best] = EI / P(f(x) < f_best)
    # This is the conditional expected improvement.
    # To avoid division by zero when cdf is very small, we use a small epsilon
    # But we need to be careful: if cdf is very small, the conditional expected improvement
    # can be very large, which might lead to exploring very uncertain regions.
    # We clip the cdf to avoid extreme values.
    cdf_clipped = np.clip(cdf, 1e-6, 1.0)
    
    # QI score: conditional expected improvement
    qi = ei / cdf_clipped
    
    # Cost-aware score: divide by cost to account for expense
    # Higher score is better
    scores = qi / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
