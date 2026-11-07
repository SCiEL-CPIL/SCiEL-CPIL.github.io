"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track global start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to leave 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Samples a value from the GP posterior N(mu, sigma^2) for each candidate.
    Returns negative sampled values so that the candidate with the lowest
    predicted function value has the highest score (since the evaluator
    picks the candidate with the highest score).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far (not used in standard TS,
        kept for interface compatibility).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores where higher means evaluate next.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: use LCB with small kappa for speed
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is positive and numerically stable
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw one sample from N(mu, sigma^2) for each candidate
    # z ~ N(0, 1)
    z = np.random.standard_normal(size=mu.shape)
    samples = mu + sigma_safe * z
    
    # We want the candidate with the minimum sampled value to have the maximum score.
    # Since the evaluator picks the candidate with the HIGHEST score, we return
    # the negative of the sampled values.
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
