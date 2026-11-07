import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement (lower true function value).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    # 1e-8 is a safe lower bound
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the z-score for the expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a UCB-like exploration term
    # This encourages exploring high-uncertainty regions, which is beneficial
    # in high-dimensional spaces with limited samples.
    beta = 0.1
    ucb_boost = beta * sigma_safe
    
    # Combine EI and UCB boost
    scores = ei + ucb_boost
    
    # Ensure all returned scores are finite
    # Replace non-finite values with 0.0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
