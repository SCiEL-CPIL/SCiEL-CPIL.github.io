import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a time-aware exploration bonus.
    Since we only have mu and sigma for the candidates (not the full GP posterior),
    we use a standard EI formulation for minimization, plus a small exploration term
    to encourage visiting uncertain regions early on.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    # Ensure inputs are finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-10)  # avoid division by zero
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values causing numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    # EI for minimization
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Add a small exploration bonus to encourage exploring high-uncertainty regions
    # The bonus is proportional to sigma, but scaled down to not dominate the EI
    # This helps in early stages when f_best is not yet well-optimized
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
