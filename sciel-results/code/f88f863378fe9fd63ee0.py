"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    UCB acquisition for Bayesian optimization of Goldstein-Price 2D.
    
    Uses Upper Confidence Bound (UCB) for minimization.
    score = -(mu - kappa * sigma)
    This is equivalent to minimizing (mu - kappa * sigma), where:
    - mu is the predicted mean (lower is better)
    - sigma is the predicted standard deviation (uncertainty)
    - kappa controls the exploration/exploitation trade-off
    
    Higher scores indicate more promising candidates to evaluate next.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # UCB for minimization: we want to minimize, so we maximize
    # the negative of the lower confidence bound
    # Lower confidence bound: mu - kappa * sigma
    # We maximize: -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # Exploration coefficient: balance between exploitation and exploration
    # kappa = 1.0 provides a reasonable balance for this problem
    kappa = 1.0
    
    # Compute UCB scores (higher is better)
    scores = -mu + kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
