import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to prevent division by zero and numerical instability
    min_sigma = 1e-6
    sigma_safe = np.where(sigma > min_sigma, sigma, min_sigma)
    
    # Calculate the Z-scores for the potential improvement
    # z represents how many standard deviations the predicted mean is below f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate the standard normal CDF and PDF
    # Phi(z) = P(Z <= z)
    # phi(z) = standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * pdf(z)
    # The term (f_best - mu) * Phi(z) captures the expected gain where improvement is positive
    # The term sigma * phi(z) is the correction factor from the normal distribution
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
