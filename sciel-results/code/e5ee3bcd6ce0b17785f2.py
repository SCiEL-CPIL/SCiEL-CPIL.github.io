"""Rewritten acquisition function using a hybrid UCB/EI strategy with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit from task contract
_SAFETY_MARGIN = 5.0  # seconds to leave before deadline


def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _start_time)


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines UCB exploration with EI exploitation.
    
    Uses a high UCB coefficient to aggressively explore distant regions
    of the Branin function where local minima may exist, while still
    exploiting regions where improvement is likely.
    
    The score is a weighted combination of:
    - UCB term: mu + beta * sigma (exploration)
    - EI term: standard expected improvement (exploitation)
    
    We normalize both to comparable scales and blend them.
    """
    # Check time budget
    if _time_left() < 0.1:
        # Return a simple UCB score as fallback
        return mu + 5.0 * sigma

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB component: aggressive exploration
    beta_ucb = 5.0
    ucb_score = mu + beta_ucb * sigma
    
    # EI component: exploitation
    z = (f_best - mu) / sigma
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to zero mean, unit variance for blending
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std < 1e-12:
        ucb_std = 1.0
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score)
    if ei_std < 1e-12:
        ei_std = 1.0
    ei_norm = (ei_score - ei_mean) / ei_std
    
    # Blend: weight toward UCB for exploration, but keep some EI
    # The idea: high UCB weight to find distant minima, EI to refine
    w_ucb = 0.7
    w_ei = 0.3
    
    combined = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Convert back to a scale where higher is better
    # Use the original UCB scale as reference
    scores = combined * ucb_std + ucb_mean
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
