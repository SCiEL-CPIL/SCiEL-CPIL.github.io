"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Hartmann 6D.
    
    Uses a Thompson Sampling-style approach: sample a single objective value
    from the posterior at each candidate point and select the point with the
    lowest sampled value. This naturally balances exploration and exploitation
    without tuning hyperparameters like kappa or xi.
    
    For a Gaussian posterior, sampling y ~ N(mu, sigma^2) at each point and
    selecting argmin y is equivalent to maximizing -mu + sigma * z where z is
    a fixed standard normal draw. We implement this by computing a score of
    the form: -mu + sigma * z_i for a random vector z drawn from N(0, I).
    
    To make this deterministic per call (since we only get mu, sigma, f_best
    and no random state), we use a hash-based pseudo-random seed derived from
    the input arrays to generate a reproducible random draw. However, since
    the evaluator calls this function fresh each time with new mu/sigma, we
    can simply use a fixed random seed per call or generate fresh randoms.
    
    Actually, the simplest correct Thompson Sampling approach:
    score_i = mu_i - sigma_i * z_i  where z_i ~ N(0,1)
    We want to MINIMIZE the true objective, so we select the point with the
    LOWEST sampled value. But the contract says "higher meaning evaluate next".
    So we return -sampled_value, i.e., -mu + sigma * z.
    
    We draw z from a standard normal. To avoid needing a persistent RNG state,
    we generate fresh randoms each call. This is valid Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean of the GP at candidate points (in objective units, lower is better)
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation of the GP at candidate points (in objective units)
    f_best : float
        Best (lowest) observed objective value so far
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means evaluate next
    """
    # ----- time guard (safety margin) -----
    t0 = time.time()
    deadline = t0 + 295.0  # leave 5 s for caller overhead

    n = mu.shape[0]

    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)

    # Thompson Sampling: draw z ~ N(0, I_n)
    # Use numpy's default RNG for simplicity and speed
    z = np.random.standard_normal(n)
    
    # Score = -mu + sigma * z
    # This is the negative of the sampled value y = mu - sigma * z
    # We want to maximize this score to select the point with lowest y
    scores = -mu + sigma_safe * z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # If we are dangerously close to the deadline, return immediately
    if time.time() > deadline - 0.5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
