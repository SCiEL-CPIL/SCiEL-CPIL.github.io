"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB (time-varying beta) and normalized Expected Improvement acquisition function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    global _obs_count, _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe score
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return np.zeros_like(mu)
    
    t = _obs_count
    _obs_count += 1
    
    # Ensure inputs are finite
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Time-varying beta for GP-UCB: beta_t = sqrt(2*t*log(1+t))
    beta = np.sqrt(2.0 * t * np.log(1.0 + t))
    
    # GP-UCB score for minimization: maximize (-mu + sqrt(beta) * sigma)
    ucb_score = -mu + np.sqrt(beta) * sigma
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Normalize both components to [0, 1] for balanced combination
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Adaptive weighting: shift from exploration to exploitation as t grows
    # Early: more exploration (higher UCB weight), Late: more exploitation (higher EI weight)
    # Use a smooth transition over the 30-evaluation budget
    w_ucb = 0.8 - 0.3 * (t / 30.0)
    w_ei = 1.0 - w_ucb
    
    scores = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
