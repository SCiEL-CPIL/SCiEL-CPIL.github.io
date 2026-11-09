"""Cost-aware acquisition: quantile-based Expected Improvement on the
cost-normalized objective.

Structural change from UCB-per-cost: instead of dividing a pointwise UCB by
cost (a biased approximation that ignores the interaction between f-uncertainty
and cost), we model the posterior of the *ratio* R = f / cost directly.

Given f ~ N(mu, sigma^2) and known cost c, the ratio R = f/c has
E[R] = mu/c and Var[R] = sigma^2 / c^2. We compute a quantile (median)
improvement score: P(R < f_best / c) * (f_best/c - E[R]) + a small
uncertainty bonus scaled by the ratio's standard deviation. This captures
how high cost both shifts the expected normalized value and shrinks its
uncertainty, which the naive division does not.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import stats

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep the posterior meaningful when GP is
    # very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # --- Quantile-based EI on the cost-normalized objective R = f / cost ---
    #
    # R = f / c where f ~ N(mu, sigma^2) and c is known.
    # E[R] = mu / c
    # Std[R] = sigma / c
    #
    # We want to maximize the expected improvement of R over the current
    # best *normalized* value f_best / c. The exact EI for a normal
    # posterior is:
    #   EI = (f_best/c - mu/c) * Phi((f_best/c - mu/c) / (sigma/c))
    #        + (sigma/c) * phi((f_best/c - mu/c) / (sigma/c))
    #
    # This is the standard EI formula applied to the ratio's posterior.
    # It naturally accounts for both the shift (mu/c) and the shrinkage
    # of uncertainty (sigma/c) due to cost.

    # Normalize: work in units of the ratio.
    mu_r = mu / cost_safe          # E[R]
    sigma_r = sigma_safe / cost_safe  # Std[R]
    best_r = f_best / cost_safe    # current best normalized value (per-candidate)

    # Standardized distance: how many sigma_r below best_r is the mean?
    z = (best_r - mu_r) / np.maximum(sigma_r, 1e-12)

    # Clip z to avoid numerical overflow in the normal CDF/PDF.
    z_clipped = np.clip(z, -10.0, 10.0)

    # EI for minimization: E[max(best_r - R, 0)]
    # = (best_r - mu_r) * Phi(z) + sigma_r * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal.
    #
    # For minimization, improvement is positive when R < best_r.
    # EI = (best_r - mu_r) * Phi(z) + sigma_r * phi(z)
    # This is always >= 0.

    phi_z = stats.norm.pdf(z_clipped)
    Phi_z = stats.norm.cdf(z_clipped)

    ei = (best_r - mu_r) * Phi_z + sigma_r * phi_z

    # Add a small exploration bonus proportional to the ratio's std to
    # encourage visiting uncertain regions. This is a mild UCB-style
    # term on the normalized objective.
    kappa = 0.5
    exploration = kappa * sigma_r

    score = ei + exploration

    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
