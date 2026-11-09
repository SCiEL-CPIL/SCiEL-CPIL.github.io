"""Rewritten acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a linear exploration bonus.
    
    This approach avoids the instability of normalizing acquisition scores across the batch,
    which can distort relative rankings when the scale of EI or UCB changes drastically.
    Instead, it uses a stable weighted sum: EI + gamma * sigma.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Exploration bonus: proportional to uncertainty
    # A coefficient of 0.8 balances exploitation (EI) and exploration (sigma).
    # This is equivalent to a soft UCB where the exploration weight is constant.
    gamma = 0.8
    scores = ei + gamma * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
