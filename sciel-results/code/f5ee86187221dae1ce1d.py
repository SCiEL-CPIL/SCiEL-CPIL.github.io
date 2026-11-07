"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    lower-confidence-bound (LCB) style exploration term.
    
    Parameters:
        mu: array of shape (N,) with posterior means
        sigma: array of shape (N,) with posterior standard deviations
        f_best: float, the best (lowest) observed value so far
    
    Returns:
        scores: array of shape (N,) with acquisition scores (higher = better)
    """
    # Safety: handle edge cases
    if sigma is None or len(sigma) == 0:
        return np.zeros(len(mu))
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB term: encourages exploration in uncertain regions
    # Use a moderate exploration coefficient
    xi = 0.5  # exploration-exploitation trade-off
    lcb = -xi * sigma
    
    # Combine EI and LCB: EI drives exploitation, LCB drives exploration
    # Normalize both to have similar scales before combining
    # Actually, just use a weighted sum. EI is typically the dominant term.
    # A common approach: use EI as primary, add a small LCB component
    scores = ei + 0.3 * lcb
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
