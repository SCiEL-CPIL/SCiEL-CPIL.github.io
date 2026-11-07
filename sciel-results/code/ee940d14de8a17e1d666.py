import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function for minimization.
    
    POI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    This is a bounded [0, 1] score that emphasizes candidates likely to
    improve over the current best, providing a different exploration-
    exploitation balance than EI or UCB.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Avoid division by zero and negative sigma
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Z-score: Z = (f_best - mu) / sigma
    # Higher Z means more likely to improve
    Z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: POI = Phi(Z)
    POI = norm.cdf(Z)
    
    # Ensure values are in [0, 1] and finite
    POI = np.clip(POI, 0.0, 1.0)
    POI = np.where(np.isfinite(POI), POI, 0.0)
    
    return POI
