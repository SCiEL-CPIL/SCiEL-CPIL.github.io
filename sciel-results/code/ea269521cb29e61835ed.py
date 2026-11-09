"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Gaussian Process Lower Confidence Bound (GP-LCB) acquisition function.
    
    Args:
        mu: Array of mean predictions (shape: (2000,))
        sigma: Array of standard deviation predictions (shape: (2000,))
        f_best: The best observed value so far (scalar)
        
    Returns:
        Array of acquisition scores (shape: (2000,)), higher is better.
    """
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # GP-LCB: score = -mu - kappa * sigma
    # We maximize this score.
    # kappa controls the balance between exploitation (low mu) and exploration (high sigma).
    kappa = 2.0
    
    scores = -mu - kappa * sigma
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
