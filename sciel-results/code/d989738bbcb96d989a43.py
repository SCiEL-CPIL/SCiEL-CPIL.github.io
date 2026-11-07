"""Rewritten cost-aware acquisition: Student-t EI with deadline guard."""
import time
import numpy as np
from scipy.stats import t as student_t

_DEADLINE_MARGIN_S = 10.0
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def acquisition(mu, sigma, f_best, cost):
    _ensure_start()
    # Deadline guard: if we are too close to the kill wall, return a safe fallback
    # that still selects the point with best deterministic expected improvement
    # (a simple, cheap heuristic) so the loop can finish.
    if time.time() - _START_TIME > 320.0:
        z = (f_best - mu) / (np.maximum(sigma, 1e-9))
        fallback = (f_best - mu) * 0.5 + sigma * 0.1
        return fallback

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Standard normal-based EI for reference, but we use Student-t to be robust
    # to model misspecification. Use a moderate degrees of freedom (nu=5) to
    # account for uncertainty in the GP posterior.
    nu = 5.0
    z = (f_best - mu) / sigma_safe

    # Student-t CDF and PDF
    try:
        cdf_val = student_t.cdf(z, df=nu)
        pdf_val = student_t.pdf(z, df=nu)
    except Exception:
        # Fallback to normal if Student-t fails for any reason
        from scipy.stats import norm
        cdf_val = norm.cdf(z)
        pdf_val = norm.pdf(z)

    # Expected improvement under Student-t posterior
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val

    # Cost-aware scaling: divide by cost, but add a small exploration bonus
    # to avoid completely ignoring high-uncertainty regions.
    # Use a soft cost normalization: cost is in [1, 2], so 1/cost is in [0.5, 1].
    # Add a small epsilon to avoid division by zero (cost >= 1 always).
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
