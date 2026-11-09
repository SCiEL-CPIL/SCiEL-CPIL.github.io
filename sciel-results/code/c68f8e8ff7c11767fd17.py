import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Dynamic xi based on current best
    # If f_best is large, we want larger improvements.
    # If f_best is close to 0, we want small improvements.
    xi = 0.05 * f_best
    
    # Calculate z-score
    # sigma can be very small, add epsilon
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - xi - mu) / sigma_safe
    
    # Expected Improvement with xi
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # Note: if f_best - xi - mu is negative, EI is 0?
    # The formula (f_best - xi - mu) * Phi(z) + sigma * phi(z) works for all z.
    # If z is very negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0.
    # If z is very positive, Phi(z) ~ 1, EI ~ f_best - xi - mu.
    
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical errors might cause small negatives)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score
    # cost is (2000,)
    # Avoid division by zero (cost >= 1)
    score = ei / cost
    
    return score
