import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a small
    exploration term to improve search in multimodal landscapes.
    
    Args:
        mu: Array of posterior means, shape (2000,)
        sigma: Array of posterior standard deviations, shape (2000,)
        f_best: The best (minimum) objective value observed so far.
        
    Returns:
        Array of acquisition scores, shape (2000,), higher is better.
    """
    # Ensure sigma is positive and finite to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite; replace non-finite with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in cdf/pdf
    # norm.cdf and norm.pdf are stable, but clipping adds a layer of safety
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting high-uncertainty regions.
    # This helps escape local minima in the RBF mixture.
    # A small coefficient (0.1) ensures exploitation remains dominant.
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite; replace non-finite with 0.0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
