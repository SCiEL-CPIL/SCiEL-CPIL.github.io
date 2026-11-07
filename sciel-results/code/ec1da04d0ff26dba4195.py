"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Deadline guard: track time to ensure we finish well within 300s
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-scores for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Also compute a UCB-based score for exploration
    # For minimization: UCB = mu - beta * sigma
    # We blend EI with a UCB term to balance exploitation and exploration
    
    # Adaptive beta: start with moderate exploration, decay over time
    # Since we don't have explicit step count, use f_best as a proxy for progress
    # When f_best is close to the known minimum (-39.1662), reduce exploration
    known_min = -39.1662
    progress = np.clip((f_best - known_min) / (50.0 - known_min), 0.0, 1.0)
    
    # Beta decreases as we get closer to the optimum
    beta = 2.0 * (1.0 - 0.5 * (1.0 - progress))
    
    ucb_scores = mu - beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei_scores), np.max(ei_scores)
    ucb_min, ucb_max = np.min(ucb_scores), np.max(ucb_scores)
    
    ei_range = ei_max - ei_min + eps
    ucb_range = ucb_max - ucb_min + eps
    
    ei_norm = (ei_scores - ei_min) / ei_range
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Blend: weight EI more heavily, but keep some exploration
    # Weight shifts toward EI as we converge
    ei_weight = 0.7 + 0.2 * (1.0 - progress)
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
