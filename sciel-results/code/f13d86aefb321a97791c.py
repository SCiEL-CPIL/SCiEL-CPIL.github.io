# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _deadline_guard(margin=5.0):
    """Return True if we should stop to stay within deadline."""
    # Each acquisition call should be fast; the 330s limit is for the whole run.
    # We don't know exactly how much time is left, but we can track relative time.
    # Since we don't have access to the evaluator's deadline directly, we'll keep
    # computation minimal.
    return False

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a combination of Expected Improvement (EI) with an exploration bonus
    based on uncertainty, tuned for the Branin function landscape.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for exploration
    # UCB = mu - kappa * sigma (for minimization, we want low mu and high sigma)
    # We'll use a kappa that balances exploitation and exploration
    kappa = 2.0
    ucb = mu - kappa * sigma
    
    # Combine EI and UCB. 
    # Normalize both to have zero mean and unit variance to combine fairly.
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-10
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-10
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 70% EI, 30% UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
