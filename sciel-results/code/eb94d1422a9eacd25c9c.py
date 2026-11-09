import numpy as np
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function: Expected Improvement with exploration adjustment.
    """
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement using stable formulas
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    Phi_z = ndtr(z)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Base EI
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Add a small exploration term to encourage diversity
    # This helps when EI is very concentrated
    exploration_weight = 0.05
    exploration = exploration_weight * sigma
    
    scores = ei + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
