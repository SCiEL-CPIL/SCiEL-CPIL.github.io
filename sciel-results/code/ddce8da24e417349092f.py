"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and an Upper Confidence Bound (UCB) exploration term, normalized by cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) objective value observed so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term
    kappa = 1.5
    ucb = mu + kappa * sigma_safe
    
    # Combine EI and UCB: use EI as primary exploitation signal,
    # add a scaled UCB bonus to encourage exploration in high-uncertainty regions.
    # Normalize ucb relative to f_best to make it comparable to EI scale.
    ucb_normalized = np.maximum(f_best - ucb, 0.0)  # positive when ucb < f_best
    # Actually for minimization, UCB should be low; we want to explore where
    # the lower bound of the GP (mu - kappa*sigma) is promising.
    # A better exploration term: the potential improvement from uncertainty
    # is captured by sigma itself scaled appropriately.
    
    # Alternative: blend EI with a pure uncertainty bonus
    # score = (ei + beta * sigma) / cost
    # where beta controls exploration-exploitation tradeoff
    
    beta = 0.5
    score = (ei + beta * sigma_safe) / np.maximum(cost, 1e-8)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
