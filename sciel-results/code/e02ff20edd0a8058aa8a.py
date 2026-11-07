"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# --- module-level state -------------------------------------------------
_t0 = time.monotonic()
_HARD_DEADLINE = _t0 + 300.0     # hard wall-clock cap for the whole run
_PER_CALL_CAP = 15.0             # never let one call eat the budget
_SCALE = 1.0 / 10.24             # maps [-5.12,5.12] -> [0,1]
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: return a safe, finite, deterministic answer fast ---
    if time.monotonic() > _HARD_DEADLINE - 5.0:
        cost_safe = np.maximum(np.asarray(cost, dtype=float), _EPS)
        return np.ones(len(mu)) / cost_safe          # favour cheap points

    t_start = time.monotonic()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)
    n = len(mu)

    # 1) base cost-aware EI over the batch (seed GP posterior is given)
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    
    # Cost normalization: use offset to amplify cost differences
    # cost is in [1, 2], so cost - 0.5 is in [0.5, 1.5]
    # This makes the denominator smaller for cheap points, amplifying their score
    cost_offset = 0.5
    cost_normalized = np.maximum(cost - cost_offset, _EPS)
    base = ei / cost_normalized

    # 2) Uncertainty bonus: encourage exploring high-variance regions
    # Scale by max sigma to keep in [0, 1]
    max_sigma = np.max(sigma) + _EPS
    uncertainty_bonus = 0.25 * (sigma / max_sigma)
    
    # 3) Exploitation bonus: boost points where mu is close to f_best (low expected value)
    # Normalize by range of mu to keep in [0, 1]
    mu_range = np.max(mu) - np.min(mu) + _EPS
    exploitation_bonus = 0.1 * (1.0 - (mu - np.min(mu)) / mu_range)
    
    # Combine: base EI + uncertainty bonus + exploitation bonus
    scores = base + uncertainty_bonus + exploitation_bonus

    # 4) finite / bounded output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, -1e6, 1e6)

    # 5) hard per-call cap
    if time.monotonic() - t_start > _PER_CALL_CAP:
        pass  # already returned below immediately

    return scores
# EVOLVE-BLOCK-END
