"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Computes cost-aware Expected Improvement scores.
    
    Args:
        mu: Array of shape (2000,) containing mean predictions.
        sigma: Array of shape (2000,) containing standard deviations.
        f_best: Scalar or array containing the best observed value.
        cost: Array of shape (2000,) containing the cost of evaluation.
        
    Returns:
        Array of shape (2000,) with acquisition scores.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # f_best is likely a scalar, but we use broadcasting just in case
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware adjustment
    # Subtracting a small multiple of the cost to balance exploration and cost
    # Based on knowledge that lambda=0.1 performed well
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
