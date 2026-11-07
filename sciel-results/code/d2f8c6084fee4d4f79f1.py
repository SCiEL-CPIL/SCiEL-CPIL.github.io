import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Deterministic Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), acquisition values (higher is better)
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to find points with low mu and high sigma.
    # Score = -mu - kappa * sigma
    # Higher score means lower predicted value with larger uncertainty penalty (optimistic for finding minima).
    kappa = 1.5
    
    scores = -mu - kappa * sigma_safe
    
    # Ensure finiteness as required by the contract
    # If any non-finite values appear, replace with very low scores to avoid selection
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
