"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement with adaptive exploration.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Upper Confidence Bound (UCB) for exploration
    3. Adaptive weighting based on the gap between f_best and the GP mean
    
    Parameters:
    -----------
    mu : np.ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) objective value observed so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (n,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Safety: ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Handle degenerate sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    # UCB for minimization: -mu + beta * sigma
    # Adaptive beta: larger when we've made progress, smaller when close to optimum
    # Scale based on how far f_best is from typical GP predictions
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-10
    
    # Normalize sigma to [0, 1] range for adaptive beta
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Adaptive exploration coefficient
    # When f_best is much lower than most GP means, exploit more
    # When f_best is close to GP means, explore more
    gap = f_best - mu_min
    gap_scale = np.abs(gap) / (mu_range + 1e-10)
    
    # Beta decreases as we get closer to the optimum (smaller gap)
    # Beta is between 0.3 and 1.5
    beta = 0.3 + 1.2 * np.clip(1.0 - gap_scale, 0.0, 1.0)
    
    # UCB scores (higher = more exploration)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to [0, 1] for combining
    ei_max = np.max(ei) + 1e-10
    ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb) - np.min(ucb) + 1e-10
    ucb_norm = (ucb - np.min(ucb)) / ucb_max
    
    # Weight: more EI when exploitation is strong, more UCB when exploring
    # Use the gap to determine weighting
    # If gap is large (we're far from optimum), explore more
    # If gap is small (we're close), exploit more
    ei_weight = 0.6 + 0.4 * np.clip(1.0 - gap_scale, 0.0, 1.0)
    ucb_weight = 1.0 - ei_weight
    
    # Combined score
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
