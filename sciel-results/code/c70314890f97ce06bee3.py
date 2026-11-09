"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative cost gating.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values from numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Calibrate k to the expected EI scale
    # Use the median of positive EI values as a proxy for the typical improvement scale.
    positive_ei = ei[ei > 1e-8]
    if len(positive_ei) > 0:
        ei_scale = np.median(positive_ei)
    else:
        ei_scale = 1.0
    
    # k is inversely proportional to the EI scale.
    # If EI is large, k is small (cost penalty is less severe).
    # If EI is small, k is large (cost penalty is more severe).
    # We use raw cost (1 to 2) as per directive.
    k = 1.0 / (ei_scale + 1e-6)
    
    # Final score: EI * exp(-k * cost)
    # Note: Using raw cost as directed.
    scores = ei * np.exp(-k * cost)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
