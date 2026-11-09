"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement + uncertainty-driven exploration.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.clip(ei, -1e6, 1e6)

    # Scale reference: use f_best magnitude but floor at 1.0 for stability
    scale = max(abs(f_best), 1.0)

    # UCB-style exploration: balance exploitation and exploration
    # beta adapts to the ratio of typical sigma to typical improvement
    sigma_median = np.median(sigma_safe)
    ei_median = np.median(np.maximum(ei, 1e-12))
    
    # beta controls exploration strength; larger when sigma is large relative to EI
    beta = 0.5 * (sigma_median / (ei_median + 1e-12))
    beta = np.clip(beta, 0.1, 3.0)
    
    ucb = -mu + beta * sigma_safe  # higher is better for UCB (minimize -mu + beta*sigma)
    
    # Normalize both EI and UCB to comparable scales
    # Use robust scaling: shift by median, scale by IQR
    ei_med = np.median(ei)
    ei_iqr = np.percentile(ei, 75) - np.percentile(ei, 25)
    if ei_iqr < 1e-12:
        ei_iqr = 1e-12
    ei_scaled = (ei - ei_med) / ei_iqr
    
    ucb_med = np.median(ucb)
    ucb_iqr = np.percentile(ucb, 75) - np.percentile(ucb, 25)
    if ucb_iqr < 1e-12:
        ucb_iqr = 1e-12
    ucb_scaled = (ucb - ucb_med) / ucb_iqr
    
    # Blend EI and UCB with emphasis on EI
    w_ei = 0.70
    w_ucb = 0.30
    scores = w_ei * ei_scaled + w_ucb * ucb_scaled
    
    # Stronger penalty for candidates whose mean is well above f_best
    # This prevents wasting evaluations on obviously poor regions
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 5.0)
    scores = scores - 0.15 * mu_penalty
    
    # Additional penalty for very high sigma relative to f_best (diminishing returns)
    sigma_penalty = np.maximum(0.0, (sigma_safe - 0.5 * scale) / (scale + 1e-12))
    sigma_penalty = np.minimum(sigma_penalty, 2.0)
    scores = scores - 0.05 * sigma_penalty

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard
    if time.time() - t0 > deadline:
        pass  # already computed; return immediately

    return scores
# EVOLVE-BLOCK-END
