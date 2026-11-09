"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# State tracking with seed detection
_state = {"call_count": 0, "last_f_best": None, "seed_start": None}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with adaptive UCB bonus.
    
    Detects new seeds by monitoring f_best jumps and resets the step counter.
    
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
    # Detect new seed: if f_best is much worse than before, it's a new random init
    if _state["last_f_best"] is not None:
        # If f_best jumped up by more than 50% of previous, likely new seed
        if f_best > _state["last_f_best"] * 1.5 + 0.1:
            _state["call_count"] = 0
            _state["seed_start"] = time.time()
    _state["last_f_best"] = f_best
    _state["call_count"] += 1
    
    # Also reset if call count exceeds expected max per seed (safety)
    if _state["call_count"] > 25:
        _state["call_count"] = 1
    
    step = _state["call_count"] - 1  # 0-indexed step
    
    # Clamp sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are CDF and PDF of standard normal
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative; clip to ensure
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB bonus for exploration
    # Early: high exploration, late: low
    beta_max = 1.5
    beta_min = 0.05
    tau = 8.0
    beta = beta_min + (beta_max - beta_min) * np.exp(-step / tau)
    
    # Combine: EI + beta * sigma (UCB-style exploration)
    scores = ei + beta * sigma_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
