# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-effective Expected Improvement.
    
    Maximizes the Expected Improvement per unit of cost to optimize
    the use of the fixed evaluation budget.
    
    Parameters
    ----------
    mu : array-like, shape (n,)
        GP mean predictions.
    sigma : array-like, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI divided by cost
    # This metric favors candidates that offer high improvement relative to their cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
