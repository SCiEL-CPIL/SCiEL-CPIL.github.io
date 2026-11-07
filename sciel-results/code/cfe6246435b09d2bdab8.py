import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) acquisition function
    # Handles cost by normalizing EI by the evaluation cost
    
    # Small epsilon to avoid division by zero in sigma
    eps = 1e-8
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of standard normal at z
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    
    # Using scipy.stats.norm for accuracy and stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip negative values due to numerical errors
    ei = np.clip(ei, 0.0, None)
    
    # Handle any remaining NaNs or Infs
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize by cost
    cost_safe = np.maximum(cost, eps)
    score = ei / cost_safe
    
    # Ensure final scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
