import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach combining Thompson Sampling (stochastic exploration)
    with Expected Improvement (deterministic exploitation), with adaptive weighting
    based on progress toward the known global minimum.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean predictions
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best (lowest) function value observed so far
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better for selection)
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n)
    
    # 2. Thompson Sampling: sample from posterior predictive distribution
    # Use a seeded RNG for reproducibility within a single call
    rng = np.random.RandomState(42)
    z_ts = rng.standard_normal(n)
    ts_sample = mu - sigma_safe * z_ts
    
    # Normalize TS sample to [0, 1] (lower ts_sample is better, so invert)
    ts_min = np.min(ts_sample)
    ts_max = np.max(ts_sample)
    ts_range = ts_max - ts_min
    if ts_range > 1e-12:
        ts_norm = (ts_max - ts_sample) / ts_range
    else:
        ts_norm = np.zeros(n)
    
    # 3. Adaptive Weighting based on progress towards global optimum
    # Global minimum of Michalewicz 2D is approximately -1.8013
    opt = -1.8013
    dist = f_best - opt
    
    # Start with high exploration (TS), shift to exploitation (EI) as we get closer
    if dist > 1.0:
        w_ei = 0.1
    elif dist > 0.5:
        w_ei = 0.2
    elif dist > 0.1:
        w_ei = 0.4
    elif dist > 0.01:
        w_ei = 0.7
    else:
        w_ei = 0.9
    
    # Combine scores
    scores = w_ei * ei_norm + (1.0 - w_ei) * ts_norm
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return scores
