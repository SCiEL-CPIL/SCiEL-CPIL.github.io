"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize inputs
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate z-score
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical instability in exp/erf
    z = np.clip(z, -10.0, 10.0)

    # Fast Expected Improvement calculation
    # cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf_z = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    # pdf(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # Hybrid acquisition: EI + small exploration bonus (sigma), all cost-aware
    # The exploration bonus helps escape local minima or unexplored regions,
    # while the cost division ensures we prefer cheap points.
    score = (ei + 0.1 * sigma) / cost

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
