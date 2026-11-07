"""Cost-aware acquisition for BO on hartmann 6d (minimisation)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _safe_norm_cdf(x):
    """Numerically stable CDF of the standard normal."""
    x = np.clip(x, -30, 30)
    return norm.cdf(x)


def _safe_norm_pdf(x):
    """Numerically stable PDF of the standard normal."""
    x = np.clip(x, -30, 30)
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Linear cost-penalized EI + UCB exploration bonus.
    
    Uses: score = EI - lambda_cost * cost + beta * sigma
    
    This is a structurally different acquisition family from EI/cost^alpha:
    - Linear penalty directly subtracts expected cost (optimal for budgeted BO)
    - UCB term adds explicit exploration proportional to uncertainty
    - Together they balance exploitation, exploration, and cost efficiency
    
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

    # UCB exploration term: beta * sigma
    # beta controls exploration-exploitation tradeoff
    # For hartmann 6d with GP, a moderate beta helps escape local optima
    beta = 1.5
    ucb = beta * sigma

    # Linear cost penalty: lambda_cost * cost
    # Cost ranges in [1, 2] (1 + mean(x) for x in [0,1]^6)
    # lambda_cost should be small relative to typical EI values
    # Hartmann 6d range is roughly [-3.3, -0.0], so EI values are O(0.1-1)
    # A small penalty encourages but doesn't dominate on cheaper points
    lambda_cost = 0.05
    cost_penalty = lambda_cost * cost

    # Combined score: exploitation (EI) + exploration (UCB) - cost
    score = ei + ucb - cost_penalty

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
