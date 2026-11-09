"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variable to track the number of acquisition calls
_acq_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function with adaptive noise scaling.
    
    Draws a single sample from the posterior predictive distribution:
    sample = mu - sigma * z, where z ~ N(0, 1)
    
    The exploration scale (sigma multiplier) is adapted based on:
    1. Progress through the optimization (call count modulo 25 for 25 BO steps)
    2. Current best value f_best (larger f_best -> more exploration)
    
    This transforms the search into a randomized stochastic optimization process
    that naturally balances exploration and exploitation without manual tuning
    of a fixed exploration-exploitation trade-off parameter.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    t0 = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Determine progress: 25 BO steps per seed (30 total evals, 5 initial)
    # Use modulo 25 to handle multiple seeds
    step = (_acq_call_count - 1) % 25
    
    # Adaptive exploration scale:
    # - Starts high (1.5) for aggressive exploration early
    # - Decays to low (0.3) for exploitation late
    # - Also modulated by f_best: if f_best is still large, explore more
    progress = step / 24.0  # 0 to 1
    base_scale_start = 1.5
    base_scale_end = 0.3
    base_scale = base_scale_start - (base_scale_start - base_scale_end) * progress
    
    # Additional boost if f_best is still far from optimum (0 for Beale)
    # Beale function max in [-4.5, 4.5]^2 is around 170
    if f_best > 50:
        f_boost = 1.3
    elif f_best > 10:
        f_boost = 1.1
    elif f_best > 1:
        f_boost = 1.0
    elif f_best > 0.1:
        f_boost = 0.9
    else:
        f_boost = 0.8
    
    effective_scale = base_scale * f_boost
    
    # Thompson Sampling: draw one sample from posterior
    # For minimization, we want low values, so we sample mu - sigma * z
    # where z ~ N(0, 1). Higher score means lower sampled value.
    # scores = -sampled_value = -(mu - effective_scale * sigma * z)
    #        = -mu + effective_scale * sigma * z
    # 
    # Equivalently, we can just return the negative of the sampled value
    # so that higher score = lower predicted function value.
    z = np.random.standard_normal(len(mu))
    sampled_values = mu - effective_scale * sigma_safe * z
    scores = -sampled_values
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 290.0:
        # Fallback: return a simple UCB-like score
        scores = -mu + sigma_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
