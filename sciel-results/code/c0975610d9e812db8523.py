"""Rewritten acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def _time_remaining(deadline, margin=5.0):
    """Return seconds remaining before deadline minus margin."""
    return deadline - time.time() - margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and Upper Confidence Bound (UCB)
    for balancing exploitation and exploration in Bayesian optimization.
    
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
    # Set a deadline guard: we want to finish well before the 300s limit per evaluation
    # But since this is called within the evaluator's loop, we need to be fast.
    # The main concern is numerical stability and speed.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    # For very large positive z, norm.cdf(z) -> 1, norm.pdf(z) -> 0
    # For very large negative z, norm.cdf(z) -> 0, norm.pdf(z) -> 0
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # UCB for exploration: mu - kappa * sigma
    # We use a moderate kappa to balance exploration
    kappa = 1.5
    ucb = mu - kappa * sigma
    
    # Combine EI and UCB: weight them to balance exploitation and exploration
    # Normalize both to have similar scales
    ei_max = np.max(ei)
    ucb_max = np.max(-ucb)  # We want to minimize ucb, so -ucb is higher for better
    
    # If ei_max is 0, all EI values are 0 (or very close), rely more on UCB
    if ei_max < 1e-12:
        scores = -ucb
    else:
        # Normalize to [0, 1] range for combination
        ei_norm = ei / ei_max
        ucb_norm = (-ucb) / (ucb_max + 1e-12) if ucb_max > 1e-12 else 0.0
        
        # Weighted combination: 0.6 * EI + 0.4 * UCB
        # This favors exploitation when there's clear improvement, but still explores
        scores = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
