import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. Uncertainty-weighted exploration term
    
    For minimization, we want to find points where the expected improvement
    over f_best is high, while also exploring uncertain regions.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means more likely to be selected
    """
    # Track time to ensure we don't exceed deadline
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    from scipy.stats import norm
    
    # Standardize
    z = (f_best - mu) / sigma
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Add an exploration term that rewards high uncertainty
    # This helps escape local optima in early iterations
    # Use a decaying exploration bonus based on sigma
    exploration_weight = 0.3
    exploration_bonus = exploration_weight * sigma
    
    # Combine EI and exploration
    scores = ei + exploration_bonus
    
    # Check time - if we've spent too long, return what we have
    if time.time() - t_start > 280:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
