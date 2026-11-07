"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for faster normal CDF/PDF
_SQRT_2 = np.sqrt(2.0)

def _norm_cdf(z):
    """Vectorized normal CDF using erf."""
    return 0.5 * (1.0 + np.erf(z / _SQRT_2))

def _norm_pdf(z):
    """Vectorized normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Refinement: tighter UCB blend, cost-aware UCB scaling (cost^0.5), faster
    norm evaluation via erf.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Core: EI (fast vectorized)
    z = (f_best - mu) / sigma_safe
    phi = _norm_cdf(z)
    pdf = _norm_pdf(z)
    ei = (f_best - mu) * phi + sigma_safe * pdf

    # UCB component with tighter exploration
    # f_best for griewank is in [0, ~0.15]; progress high => f_best large
    progress = np.clip(f_best, 0.0, 1.0)
    # Smaller beta than before; drops faster as we improve
    beta = 0.15 * (0.5 + progress)

    # UCB in objective units: prefer low mu (negative sign), add uncertainty
    ucb_raw = -mu + beta * sigma_safe

    # Cost-aware: divide by cost so expensive points are penalized,
    # but use sqrt(cost) for the UCB term so long-jump exploration
    # isn't over-suppressed
    ucb_cost_aware = ucb_raw / np.sqrt(cost_safe)

    # Scale UCB contribution relative to EI
    ei_abs_scale = np.median(np.abs(ei))
    ucb_abs_scale = np.median(np.abs(ucb_cost_aware))
    if ei_abs_scale > 1e-9 and ucb_abs_scale > 1e-9:
        # Normalize UCB to EI scale, with a modest coefficient
        ucb_scale = 0.10 * (ei_abs_scale / ucb_abs_scale)
    else:
        ucb_scale = 0.0

    score = ei / cost_safe + ucb_scale * ucb_cost_aware

    # Clamp non-finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return ei / cost_safe

    return score
# EVOLVE-BLOCK-END
