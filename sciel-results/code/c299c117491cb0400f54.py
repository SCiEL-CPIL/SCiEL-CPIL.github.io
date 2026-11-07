import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Probability of Improvement (PI) acquisition function.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,)
    f_best (float): Best observed value so far.
    
    Returns:
    scores (np.ndarray): PI scores for each candidate, shape (N,)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive. If mu > f_best, z is negative.
    z = (f_best - mu) / sigma
    
    # Probability of Improvement is P(f(x) < f_best) = Phi(z)
    # Where Phi is the CDF of the standard normal distribution
    scores = norm.cdf(z)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
