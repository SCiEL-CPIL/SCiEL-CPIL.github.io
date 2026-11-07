# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CAEI).
    
    This function computes the Expected Improvement (EI) for each candidate point
    and divides it by the evaluation cost to maximize the expected regret reduction
    per unit cost.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score(x) = EI(x) / cost(x)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # Phi(z) is the CDF, phi(z) is the PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula
    # If z is very large positive, EI ~ f_best - mu
    # If z is very large negative, EI ~ 0
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to avoid extreme values
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
# EVOLVE-BLOCK-END
