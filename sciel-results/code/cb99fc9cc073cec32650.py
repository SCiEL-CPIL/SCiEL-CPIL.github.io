# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses Expected Improvement (EI) acquisition function.
    EI is preferred for small budgets as it adapts exploration based on
    the current best value f_best.
    """
    # Ensure sigma is strictly positive to avoid division by zero in z-score
    # If sigma is 0, the point is known, so improvement is max(0, f_best - mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: how many standard deviations away from the mean
    # the current best is.
    # z = (f_best - mu) / sigma
    # If f_best > mu, z is positive, meaning the point is likely better than best.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    score = (f_best - mu) * cdf + sigma * pdf
    
    # Handle the case where sigma was effectively 0 (though we used 1e-9)
    # For very small sigma, z is huge if f_best > mu, cdf ~ 1, pdf ~ 0.
    # score ~ f_best - mu.
    # If f_best < mu, z is huge negative, cdf ~ 0, pdf ~ 0.
    # score ~ 0.
    # This matches max(0, f_best - mu).
    
    # Ensure all scores are finite and non-negative (EI is always >= 0)
    # Due to numerical precision, very small negative values might appear.
    score = np.maximum(0.0, score)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
