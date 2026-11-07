"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 285.0  # safety margin under the 300 s limit


def _improvement(mu, sigma, f_best):
    # Expected improvement (EI) of a point over f_best.
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement.
    #
    # We want to maximize expected improvement per unit cost, but we also want
    # to avoid wasting budget on points that are already very close to the best
    # observation (tiny EI) or that have very high cost. A soft, adaptive
    # exploration term keeps the score finite and well-scaled.
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite inputs.
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    ei = _improvement(mu, sigma, f_best)

    # Base cost-aware score: EI per unit cost.
    score = ei / cost

    # Add a mild exploration bonus that rewards low-cost points with
    # relatively high uncertainty, so we do not collapse onto a single
    # high-cost optimum too early. The bonus is scaled by the typical EI so
    # it does not dominate.
    typical = float(np.median(ei)) if ei.size else 0.0
    if typical > 0:
        # Exploration: uncertainty (sigma) that is large relative to the
        # current best gap, divided by cost.
        gap = float(f_best - float(np.min(mu))) if mu.size else 0.0
        gap = max(gap, 1e-9)
        explore = (sigma / gap) / cost
        # Damp exploration so it is at most ~20% of the typical EI/cost scale.
        scale = typical / float(np.median(cost)) if cost.size else 0.0
        score = score + 0.2 * scale * explore

    # Ensure finiteness.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
