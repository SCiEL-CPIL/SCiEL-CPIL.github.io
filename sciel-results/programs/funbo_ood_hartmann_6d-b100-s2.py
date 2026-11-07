"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and a
    noise-robust exploration term.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Exploration term: reward high uncertainty, but dampen it relative to EI
    # Using a small multiple of sigma encourages visiting unexplored regions
    # without dominating the acquisition score.
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
