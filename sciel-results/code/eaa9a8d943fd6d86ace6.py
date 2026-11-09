"""Rewritten program: local refinement of the Cohn-Goncalves Laguerre root construction."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0   # safety margin before the 630 s wall-clock kill
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _seed_roots():
    return [4.0 * (i + 1) for i in range(NROOTS)]

def _root_objective(r):
    """
    Well-conditioned surrogate for the refined C4 upper bound.

    The bound is controlled by how tightly the double-root Laguerre combination
    fits in the SOS cone. A compact root set (low weighted moments) lowers the
    bound, but roots must stay well-separated so the combination remains
    non-negative (positive-definite). We balance these:

      - A weighted first+second moment term (compactness, lower is better).
      - A strong inverse-gap penalty that keeps roots from clumping, which
        would destroy positivity and inflate the true bound.
      - A soft positivity penalty.

    The weights are tuned so the unconstrained minimum sits near the seed's
    arithmetic spacing (which scores 1.0), and a local search can nudge it
    slightly lower.
    """
    r = np.asarray(r, dtype=float)
    n = r.shape[0]

    # positivity penalty
    pos_pen = np.sum(np.clip(-r, 0.0, None) ** 2)

    # gap penalties: keep roots strictly increasing and well-separated.
    gaps = np.diff(r)
    # penalize small gaps heavily (clumping destroys SOS positivity)
    gap_pen = np.sum(np.clip(0.5 - gaps, 0.0, None) ** 2) * 50.0
    # mild penalty for excessively large gaps (wastes the feasible band)
    big_gap_pen = np.sum(np.clip(gaps - 12.0, 0.0, None) ** 2) * 0.01

    # weighted moment terms (kernel e^{-x} concentrates near 0)
    w = np.exp(-0.03 * r)
    base = np.sum(w * r)
    spread = np.sum(w * r ** 2)
    obj = base + 0.3 * spread

    # entropy-like packing: encourage roots to fill the band without clumping
    log_term = np.sum(np.log(r + 1e-8))
    obj += -0.01 * log_term

    obj += pos_pen + gap_pen + big_gap_pen
    return obj

def _construct_core(seed_value):
    np.random.seed(seed_value)

    best_r = None
    best_obj = np.inf

    # Primary start: the seed's arithmetic roots (known to score 1.0).
    starts = []
    starts.append(np.array(_seed_roots(), dtype=float))

    # Small perturbations around the seed to explore the local basin.
    for _ in range(20):
        jitter = np.random.normal(0.0, 1.5, NROOTS)
        cand = np.array(_seed_roots(), dtype=float) + jitter
        cand = np.maximum.accumulate(cand)
        cand = np.maximum(cand, 1e-3)
        starts.append(cand)

    # A few geometric / log-spaced starts in a band near the seed.
    for ratio in (1.05, 1.1, 1.15, 1.2):
        cand = 4.0 * (ratio ** np.arange(NROOTS))
        starts.append(cand)

    # Log-spaced starts.
    for lo, hi in [(1.0, 50.0), (2.0, 40.0), (3.0, 35.0)]:
        cand = np.exp(np.linspace(math.log(lo), math.log(hi), NROOTS))
        starts.append(cand)

    for s in starts:
        if _deadline_reached():
            break
        try:
            res = minimize(
                _root_objective,
                s,
                method='L-BFGS-B',
                bounds=[(1e-3, None)] * NROOTS,
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10},
            )
            if res.x is not None:
                # Enforce feasibility and recompute objective.
                x = np.maximum(res.x, 1e-3)
                x = np.maximum.accumulate(x)
                obj = _root_objective(x)
                if obj < best_obj:
                    best_obj = obj
                    best_r = x
        except Exception:
            continue

    # Fallback to seed if optimization failed.
    if best_r is None:
        return _seed_roots()

    # Final polish if time remains.
    if not _deadline_reached():
        try:
            res = minimize(
                _root_objective,
                best_r,
                method='L-BFGS-B',
                bounds=[(1e-3, None)] * NROOTS,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if res.x is not None:
                x = np.maximum(res.x, 1e-3)
                x = np.maximum.accumulate(x)
                obj = _root_objective(x)
                if obj < best_obj:
                    best_r = x
                    best_obj = obj
        except Exception:
            pass

    # Ensure strict positivity & increasing order.
    best_r = np.maximum(best_r, 1e-3)
    best_r = np.maximum.accumulate(best_r)
    best_r = best_r + 1e-6

    return [float(v) for v in best_r]
# EVOLVE-BLOCK-END


def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    try:
        result = _construct_core(seed_value=0)
    except Exception:
        result = _seed_roots()

    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = _seed_roots()

    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = _seed_roots()
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
    return out
