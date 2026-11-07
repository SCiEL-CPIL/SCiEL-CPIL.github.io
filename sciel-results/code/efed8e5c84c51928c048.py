import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero or negative z issues
    # sigma is a standard deviation, so it should be >= 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If f_best < mu, z is negative.
    # EI is 0 if z < 0 (no expected improvement)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For z < 0, EI is technically 0, but the formula yields negative values?
    # Let's check.
    # If z < 0, f_best - mu < 0. Phi(z) < 0.5.
    # (f_best - mu) * Phi(z) is positive (neg * neg).
    # sigma * phi(z) is positive.
    # So EI is positive?
    # No, EI is defined as E[max(0, f_best - f(x))].
    # If f(x) > f_best, improvement is 0.
    # The formula (f_best - mu) * Phi(z) + sigma * phi(z) is correct for z > 0.
    # For z < 0, the improvement is 0.
    # Actually, the formula works for all z, but if z is very negative, EI is close to 0.
    # Let's verify.
    # If mu >> f_best, z is large negative.
    # f_best - mu is large negative.
    # Phi(z) is close to 0.
    # Product is close to 0.
    # sigma * phi(z) is close to 0.
    # So EI is close to 0.
    # This is correct.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score
    # We want to maximize EI - lambda * cost
    # Let's use lambda = 1.0
    # This means we require an expected improvement of 1 unit per unit of cost.
    # Given the function scale (min -78), 1 unit is a reasonable threshold.
    score = ei - cost
    
    # Return finite scores
    # np.where to handle any potential NaNs
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
