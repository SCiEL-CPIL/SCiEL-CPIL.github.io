import numpy as np
from scipy.stats import norm
import time

# Wall-clock deadline guard (330 s kill, use 320 s safety margin)
_T0 = time.time()
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we're running out of time, return a safe finite array
    if time.time() - _T0 > _DEADLINE:
        return np.zeros_like(mu, dtype=np.float64)

    # Clamp to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Probability of Improvement (minimization): P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)

    # Cost-normalized PI score
    score = pi / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
