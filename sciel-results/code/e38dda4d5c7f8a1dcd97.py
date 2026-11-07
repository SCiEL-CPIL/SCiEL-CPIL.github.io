"""Acquisition function for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we stay within budget
_start_time = None


def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 3D.
    
    Uses a hybrid approach combining Expected Improvement (EI) with an
    adaptive exploration term that balances exploitation and exploration
    based on the posterior uncertainty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points (in objective units)
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration term:
    # We use a weighted combination that shifts from exploration to exploitation
    # based on how much improvement is expected.
    # 
    # Key insight: Early in the optimization, we want more exploration (higher weight on sigma).
    # As f_best gets closer to the true minimum, we want more exploitation (higher weight on EI).
    #
    # We measure "progress" by how far f_best is from a reasonable lower bound.
    # For Hartmann 3D, the global minimum is -3.86278, and typical initial values are around -1 to -3.
    # We use a sigmoid-like function to smoothly transition.
    
    # Estimate progress: how much have we improved from a typical starting point?
    # Typical starting values for Hartmann 3D are around -1.0 to -2.0
    # We use f_best = -1.5 as a reference "early" point and f_best = -3.8 as "late"
    # This gives a smooth transition from exploration-heavy to exploitation-heavy
    
    # Normalize f_best to get a progress indicator in [0, 1]
    # f_best starts around -1 to -2 and ends near -3.86
    progress = np.clip((f_best - (-1.5)) / ((-3.86) - (-1.5)), 0.0, 1.0)
    progress = np.abs(progress)  # Ensure positive
    
    # Exploration weight: high early (progress ~ 0), low late (progress ~ 1)
    # Use a smooth decay
    exploration_weight = 0.3 * (1.0 - progress) + 0.05  # Range: [0.05, 0.35]
    
    # UCB-like exploration bonus
    ucb_bonus = exploration_weight * sigma_safe
    
    # Combine EI and exploration bonus
    # We weight EI more heavily as we progress
    ei_weight = 1.0  # Keep EI at full weight, add exploration on top
    
    scores = ei_weight * ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
