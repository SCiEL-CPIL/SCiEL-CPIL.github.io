"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters:
    mu : array-like, shape (2000,)
        Posterior mean of the GP.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    scores : array-like, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    # Ensure sigma is non-negative to handle potential numerical artifacts
    sigma = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to maximize (mu - beta * sigma)
    # This balances exploitation (low mu) and exploration (high sigma).
    beta = 2.5
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
