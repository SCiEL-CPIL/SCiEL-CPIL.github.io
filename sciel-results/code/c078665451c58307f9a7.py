"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses Expected Improvement (EI) as the primary acquisition signal, which is
    well-calibrated for GP-based BO. A small scale-aware exploration bonus is
    added to encourage visiting high-uncertainty regions in early stages,
    while the EI term naturally dominates as uncertainty decreases.
    
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
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is already in objective units and naturally balances exploration/exploitation
    # When sigma is large, EI has a significant contribution from sigma*phi(z)
    # When mu < f_best, EI is large due to the (f_best - mu)*Phi(z) term
    
    # Add a small exploration bonus that is scale-aware
    # The bonus should be small relative to typical EI values to not distort the ranking
    # Use a fraction of the mean EI as a scale reference
    ei_mean = np.mean(ei)
    if ei_mean > 1e-10:
        # Exploration bonus: small weight on sigma, scaled by EI magnitude
        # This ensures the bonus is meaningful in early stages (high sigma)
        # but negligible in late stages (low sigma, high EI)
        exploration_weight = 0.05 * (ei_mean / (np.max(sigma_safe) + 1e-10))
        exploration_weight = min(exploration_weight, 0.1)  # Cap at 10%
        exploration_bonus = exploration_weight * sigma_safe
    else:
        # If EI is very small, use a fixed small exploration weight
        exploration_bonus = 0.01 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
