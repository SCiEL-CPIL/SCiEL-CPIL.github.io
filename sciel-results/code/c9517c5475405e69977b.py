"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimization.
    
    Samples objective values from the GP posterior and scores candidates
    by their expected improvement over f_best. This stochastic mechanism
    naturally balances exploration and exploitation without hand-tuned
    exploration constants.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 325.0  # Leave 5s margin before 330s kill
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Use a deterministic seed based on f_best to ensure reproducibility
    # within a call, but different across iterations as f_best changes
    seed = int(abs(hash(f_best)) % (2**32))
    rng = np.random.default_rng(seed)
    
    # Thompson Sampling: sample from the GP posterior
    samples = mu + sigma_safe * rng.standard_normal(mu.shape)
    
    # For minimization: score = how much better the sample is than f_best
    # Higher score = more promising
    ts_scores = f_best - samples
    
    # Also compute Expected Improvement as a secondary signal
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip for numerical stability
    ei = np.clip(ei, -1e6, 1e6)
    ts_scores = np.clip(ts_scores, -1e6, 1e6)
    
    # Normalize both to comparable scales
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ts_max = np.max(ts_scores)
    ts_min = np.min(ts_scores)
    ts_range = ts_max - ts_min
    if ts_range > 1e-12:
        ts_norm = (ts_scores - ts_min) / ts_range
    else:
        ts_norm = np.zeros_like(ts_scores)
    
    # Blend: primarily Thompson Sampling (stochastic exploration)
    # with a smaller EI component for exploitation stability
    # TS naturally favors high-uncertainty regions when samples are low,
    # which is ideal for Beale's narrow valley structure
    scores = 0.75 * ts_norm + 0.25 * ei_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() - t0 > deadline:
        # Fallback to pure EI if we're running out of time
        scores = ei
    
    return scores
# EVOLVE-BLOCK-END
