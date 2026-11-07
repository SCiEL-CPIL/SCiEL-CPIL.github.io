"""Cost-aware acquisition for BO on hartmann 6d (minimisation)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_norm_cdf(x):
    """Numerically stable CDF of the standard normal."""
    # For very large positive x, cdf -> 1; for very large negative x, cdf -> 0
    x = np.clip(x, -30, 30)
    return norm.cdf(x)


def _safe_norm_pdf(x):
    """Numerically stable PDF of the standard normal."""
    x = np.clip(x, -30, 30)
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) per unit cost^alpha.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value (minimisation).
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x) for x in [0,1]^6.
    
    Returns
    -------
    ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    t0 = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-10)

    # z-score for EI (minimisation: improvement = f_best - mu)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Expected Improvement (for minimisation)
    cdf_z = _safe_norm_cdf(z)
    pdf_z = _safe_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI / cost^alpha
    # Use alpha = 1.0 (standard cost-aware EI). 
    # The cost is 1 + mean(x), ranging in [1, 2].
    # A moderate exponent balances exploration vs. cost.
    alpha = 1.0
    cost_penalty = np.power(cost, alpha)
    score = ei / cost_penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
