# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Cost-Aware Expected Improvement (CAEI).
    
    This approach calculates the Expected Improvement (EI) of each candidate
    point relative to the current best observed value (f_best) and normalizes
    it by the evaluation cost. This directly optimizes for the expected
    reduction in objective value per unit cost.
    
    The Expected Improvement for minimization is:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    The Cost-Aware score is:
    score = EI(x) / cost(x)
    
    This is a standard and robust method for cost-aware Bayesian optimization,
    differing from LCB by integrating over the distribution of potential
    improvements rather than using a linear bound.
    """
    # Ensure sigma is non-negative to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized improvement z
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # The formula above is standard for minimization EI.
    
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical errors might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Denominator: Cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: EI divided by cost
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
