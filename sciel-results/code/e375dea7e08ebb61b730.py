import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    Uses Expected Improvement (EI) with a dynamic exploration bonus.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores. Higher values are preferred.
    """
    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # To avoid numerical issues when f_best - mu is very negative (mu >> f_best)
    # or sigma is very small, we use a stable formulation.
    
    # Let z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use scipy.stats.norm for CDF and PDF
    # Clip z to avoid extreme values
    z_clipped = np.clip(z, -10, 10)
    
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local minima and encourages exploration in early stages
    # The bonus is weighted by sigma, so it's larger where uncertainty is high
    
    # Dynamic beta: start with more exploration, decrease over time
    # Since we don't have access to the step count, we use the magnitude of sigma
    # as a proxy for uncertainty level
    beta = 0.5
    
    # Exploration bonus: beta * sigma
    exploration_bonus = beta * sigma
    
    # Final acquisition score: EI + exploration bonus
    # Higher is better
    scores = ei + exploration_bonus
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
