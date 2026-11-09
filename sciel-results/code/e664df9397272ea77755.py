"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition with adaptive exploration.
    Samples from the GP posterior and scores by negative sampled value,
    with an additional uncertainty bonus that decays as f_best approaches 0.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    start_time = time.time()
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample one objective value per candidate from N(mu, sigma^2)
    # Use a fixed seed per call? No - we want stochasticity. Use default RNG.
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # Score: negative sampled value (lower sampled value = higher score)
    scores = -samples
    
    # Add an adaptive uncertainty bonus to encourage exploration early
    # Weight decays smoothly as f_best approaches 0
    f_best_clipped = max(f_best, 1e-6)
    # Smooth saturating decay: weight = base + amp * f_best / (f_best + half_life)
    base_weight = 0.05
    amp_weight = 0.30
    half_life = 3.0
    exploration_weight = base_weight + amp_weight * f_best_clipped / (f_best_clipped + half_life)
    
    scores += exploration_weight * sigma_safe
    
    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
