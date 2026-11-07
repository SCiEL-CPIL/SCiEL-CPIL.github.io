"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We use 320s as a safety margin.
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    Args:
        mu: Posterior mean at candidate points (shape 2000,).
        sigma: Posterior standard deviation at candidate points (shape 2000,).
        f_best: Lowest observed value so far.
        
    Returns:
        Scores for the candidate points (shape 2000,). Higher is better.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return -mu

    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # We add a small epsilon to sigma to prevent division by zero
    epsilon = 1e-9
    safe_sigma = np.maximum(sigma, epsilon)
    
    z = (f_best - mu) / safe_sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + safe_sigma * pdf_z
    
    # Ensure scores are finite and non-negative (EI can be numerically negative due to precision, but theoretically >= 0)
    # Clipping to 0 is safe as negative EI means no improvement expected.
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
