"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _call_count, _start_time
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1

    # Deadline guard: return safe default if close to 330s limit
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Estimate remaining budget fraction
    total_budget = 30.0
    avg_cost = 1.5
    max_evals = max(1, int(total_budget / avg_cost))
    total_evals = 5 + _call_count
    remaining_frac = max(0.0, 1.0 - total_evals / max_evals)

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)

    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive exploration: more exploration early in the search
    kappa = 0.05 + 0.15 * remaining_frac

    score = ei / cost_safe + kappa * sigma_safe / cost_safe

    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
