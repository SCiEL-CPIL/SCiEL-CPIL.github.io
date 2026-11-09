"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid approach that combines
    Expected Improvement (EI) with a rank-based exploration bonus.
    
    The key structural change: instead of blending normalized scores,
    we use EI as the primary signal and add a small exploration bonus
    that is proportional to the rank of sigma (uncertainty). This
    provides a more principled exploitation signal while maintaining
    exploration pressure in high-uncertainty regions.
    
    For Branin 2D with its multimodal landscape, EI naturally balances
    exploration and exploitation through the (f_best - mu)/sigma term.
    The rank-based bonus ensures we still explore regions with high
    uncertainty even when EI is low, but in a way that doesn't dominate
    the exploitation signal.
    """
    t_start = time.time()
    deadline = t_start + 295  # safety margin before 300s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Rank-based exploration bonus
    # Convert sigma to ranks in [0, 1] - higher sigma gets higher rank
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / max(n - 1, 1)
    
    # The exploration bonus is small (0.1 * max_ei) to not dominate EI
    # but provides meaningful exploration pressure
    max_ei = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    exploration_bonus = 0.1 * max_ei * sigma_rank
    
    # Combine: EI + small exploration bonus
    score = ei + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
