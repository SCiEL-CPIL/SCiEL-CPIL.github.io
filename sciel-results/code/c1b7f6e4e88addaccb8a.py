"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 292.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)
        c = np.asarray(cost, dtype=np.float64)
        s = np.maximum(s, 1e-9)
        z = np.clip((f_best - m) / s, -10, 10)
        ei = (f_best - m) * norm.cdf(z) + s * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        c = np.maximum(c, 1e-9)
        return np.where(np.isfinite(ei / c), ei / c, 0.0)

    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)

    s = np.maximum(s, 1e-9)
    c = np.maximum(c, 1e-9)

    # --- Structural change: Expected Hypervolume Improvement (EHVI) surrogate ---
    # Instead of scalar EI / cost, we approximate the expected improvement over
    # a small set of "reference" values (a grid around f_best) and weight by 1/cost.
    # This captures the shape of the posterior more faithfully than scalar EI,
    # especially when sigma varies across candidates.

    # Build a small set of reference levels: f_best + k * delta
    # Use 5 reference points spanning a range that covers most of the posterior mass
    delta = max(0.01, 0.05 * abs(f_best))
    refs = f_best + np.array([0.0, delta, 2 * delta, 3 * delta, 4 * delta])

    # For each candidate i and each reference r, compute EI_r = E[max(r - f_i, 0)]
    # where f_i ~ N(m_i, s_i^2)
    # EI_r = (r - m_i) * Phi((r - m_i)/s_i) + s_i * phi((r - m_i)/s_i)

    # shapes: refs (5,), m (2000,), s (2000,)
    # z shape: (2000, 5)
    z = (refs[None, :] - m[:, None]) / s[:, None]
    z = np.clip(z, -10, 10)
    ei = (refs[None, :] - m[:, None]) * norm.cdf(z) + s[:, None] * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # EHVI surrogate: sum over references of EI_r, with decreasing weight for farther refs
    # Use equal weights (simple, fast)
    ehvi = ei.sum(axis=1)

    # Cost-aware: divide by cost
    scores = ehvi / c

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
