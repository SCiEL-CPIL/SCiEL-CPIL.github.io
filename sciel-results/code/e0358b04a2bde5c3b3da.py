"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z_clipped = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Add a small UCB-like exploration term to encourage exploration
    # This helps escape local optima in Ackley's function
    xi = 0.01  # small exploitation bias
    beta = 0.1  # exploration weight
    
    # UCB: mu - beta * sigma (for minimization, lower is better, so we want to explore where sigma is high)
    # But we want higher score = more promising, so:
    # score = -mu + beta * sigma (high score when mu is low or sigma is high)
    # Actually, let's use a hybrid: EI + small exploration bonus
    
    # Exploration bonus: reward high uncertainty
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and exploration
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
