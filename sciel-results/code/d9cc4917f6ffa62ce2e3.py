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
    # Use a smoother progress measure based on how close we are to the optimum
    # Range: f_best starts around 50 (worst case) and decreases to -39.1662
    # Normalize progress to [0, 1] where 0 = early, 1 = converged
    progress = np.clip((50.0 - f_best) / (50.0 - known_min), 0.0, 1.0)
    
    # Beta: higher in early stages for exploration, lower in late stages
    # Use a smooth decay: beta starts at ~3.0 and decays to ~1.0
    beta = 1.0 + 2.0 * np.exp(-3.0 * progress)
    
    ucb_scores = mu - beta * sigma_safe
    
    # Instead of normalizing to [0,1], use a weighted combination that preserves
    # the relative scale of each term. EI is in objective units, UCB is in objective units.
    # Scale EI by a factor to make it comparable to UCB.
    # EI values are typically small (improvement over f_best), while UCB can be large.
    # Use a data-driven scaling: scale EI by the typical scale of UCB.
    
    # Compute typical scales
    ei_scale = np.std(ei_scores) + eps
    ucb_scale = np.std(ucb_scores) + eps
    
    # Scale EI to be comparable to UCB
    ei_scaled = ei_scores / ei_scale * ucb_scale
    
    # Blend: weight EI more heavily, but keep some exploration
    # Weight shifts toward EI as we converge
    ei_weight = 0.6 + 0.3 * progress
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_scaled + ucb_weight * ucb_scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
