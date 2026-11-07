"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = 330.0  # wall-clock seconds
_START_TIME = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Gaussian Process Lower Confidence Bound (GP-LCB).
    For minimization, we maximize the score: Score = -mu + beta * sigma.
    Beta is adapted based on the current uncertainty level (sigma).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed - 2.0  # safety margin

    if remaining <= 0:
        # Emergency: return simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Estimate progress based on uncertainty
    # Initial sigma is typically around 1.0 for standardized GP.
    # As we gather data, sigma decreases.
    sigma_max = np.max(sigma)
    
    # Map sigma_max to progress [0, 1]
    # If sigma_max is ~1.0, progress is 0 (early)
    # If sigma_max is ~0.0, progress is 1 (late)
    # Use a soft clamp to handle cases where sigma might be slightly > 1 or < 0
    progress = 1.0 - np.clip(sigma_max, 0.0, 1.0)
    
    # Beta schedule:
    # Start with high exploration (beta=5.0) to cover the space.
    # End with low exploration (beta=0.5) to exploit.
    beta_max = 5.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # GP-LCB Score for minimization:
    # We want to find points where the lower bound is low.
    # Lower Bound L(x) = mu(x) - beta * sigma(x)
    # We return scores where HIGHER is better.
    # So Score = -L(x) = -mu(x) + beta * sigma(x)
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
