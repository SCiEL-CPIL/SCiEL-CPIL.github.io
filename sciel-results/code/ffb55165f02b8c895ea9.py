import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute a GP-Lower Confidence Bound (GP-LCB) acquisition function.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        GP-LCB scores, shape (N,), higher is better.
    """
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # GP-LCB for minimization: we want to minimize f(x).
    # The lower confidence bound is LCB(x) = mu(x) - kappa * sigma(x).
    # We want to find the point with the lowest possible value, so we maximize -LCB(x).
    # score = -mu(x) + kappa * sigma(x)
    
    # Use a fixed exploration parameter
    kappa = 2.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
