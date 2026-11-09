import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coordinates)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is a structural change from Expected Improvement (EI).
    # PI focuses on the likelihood of finding a better point rather than the expected magnitude of improvement.
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-aware score: PI per unit cost
    # Higher cost means less value per unit, so we divide by cost
    # We remove the Gaussian bias towards the center and the explicit exploration bonus,
    # relying on the GP's uncertainty (sigma) via the CDF to guide exploration.
    score = pi / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
