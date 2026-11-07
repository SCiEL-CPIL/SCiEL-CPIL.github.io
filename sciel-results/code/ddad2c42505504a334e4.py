"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB and Expected Improvement.
    
    Uses a hybrid approach: primarily UCB for exploration in high dimensions,
    with a small EI component to ensure exploitation of known good regions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean predictions
    sigma : array-like, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best observed value so far (lowest)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB component: beta * sigma
    # In 6D with 30 evaluations, we need substantial exploration
    # beta=1.0 was measured to score 1.044 vs EI's 1.0
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # EI component: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine: weight UCB more heavily for exploration in high-dim
    # Normalize both to comparable scales
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    
    ucb_norm = (ucb - ucb_mean) / ucb_std
    ei_norm = (ei - ei_mean) / ei_std
    
    # Weighted combination: 70% UCB, 30% EI
    scores = 0.7 * ucb_norm + 0.3 * ei_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
