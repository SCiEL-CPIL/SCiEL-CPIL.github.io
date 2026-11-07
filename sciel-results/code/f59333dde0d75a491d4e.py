"""Cost-aware acquisition for Powell 4D BO using Expected Improvement with Gaussian cost penalty.

Structural change: replace PI-based scoring with EI (which captures magnitude of improvement,
not just probability) combined with a Gaussian decay cost penalty centered at minimum cost.
The Gaussian penalty creates a smooth, non-linear trade-off that doesn't over-suppress
moderately expensive high-improvement points, while still strongly favoring cheap evaluations.
"""
import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function with Gaussian cost penalty.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if _time_left() < 1.0:
        # Emergency fallback: just return negative mu (exploitation)
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # Gaussian decay cost penalty centered at minimum cost (1.0)
    # This creates a smooth, non-linear penalty that:
    # - Is 1.0 at the minimum cost (no penalty)
    # - Decays smoothly as cost increases
    # - Doesn't over-suppress moderately expensive points (vs. power-law)
    # The width parameter controls how quickly the penalty decays
    # Cost ranges from 1.0 to 2.0, so we tune width accordingly
    cost_min = 1.0
    cost_width = 0.35  # Controls decay rate; smaller = steeper penalty
    
    # Gaussian penalty: exp(-0.5 * ((cost - cost_min) / cost_width)^2)
    # At cost=1.0: penalty = 1.0
    # At cost=1.35: penalty = exp(-0.5) ≈ 0.607
    # At cost=1.7: penalty = exp(-0.5*(0.7/0.35)^2) = exp(-2) ≈ 0.135
    # At cost=2.0: penalty = exp(-0.5*(1.0/0.35)^2) = exp(-4.08) ≈ 0.017
    cost_penalty = np.exp(-0.5 * ((cost - cost_min) / cost_width) ** 2)

    # Combine EI with Gaussian cost penalty
    score = ei * cost_penalty

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
