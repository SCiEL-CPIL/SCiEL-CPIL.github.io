"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling variant with cost-awareness.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Scale the sampling standard deviation with the maximum uncertainty
    # This helps balance exploration across the candidate set
    max_sigma = np.max(sigma)
    
    # Sample a single scalar offset from a Gaussian distribution
    # centered at f_best with standard deviation proportional to max uncertainty
    # The scaling factor 1.0 is a reasonable default; could be tuned
    sampling_std = 1.0 * max_sigma
    offset = np.random.normal(f_best, sampling_std)
    
    # Score each candidate as -(mu - offset) / cost
    # This is equivalent to maximizing (offset - mu) / cost
    # Higher scores indicate candidates that are expected to be lower than the offset
    # and have lower cost
    scores = -(mu - offset) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
