"""Maximize product of pairwise distances for 8 planar points at fixed diameter.

Strategy: multi-stage optimization.
  1. Build a diverse pool of starting points (regular octagon, cage perturbations,
     random, and a few hand-crafted "anti-symmetric" seeds) to escape the
     regular-octagon local optimum.
  2. Polish each with L-BFGS-B on a smooth log-sum-exp surrogate of the diameter
     constraint (minimize -log(prod distances) - lambda * soft_max_log_dist^2).
  3. Differential-evolution global search over the full 16-dim space to find
     configurations in different basins.
  4. Final SLSQP polish with the exact diameter constraint, then a tight
     L-BFGS-B re-polish.
  5. Return the best valid configuration, scaled so diameter == 2.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_NPAIRS = 28
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under the 150s kill


def _elapsed():
    return time.time() - _START


def _distances(P):
    # P: (8,2) -> (28,) pairwise distances (i<j)
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    return d[np.triu_indices(_N, k=1)]


def _log_prod(P):
    d = _distances(P)
    d = np.clip(d, 1e-300, None)
    return np.sum(np.log(d))


def _diameter(P):
    return _distances(P).max()


def _score(P):
    d = _distances(P)
    dmax = d.max()
    if dmax <= 0:
        return 0.0
    ds = d / dmax
    ds = np.clip(ds, 1e-300, None)
    return (np.prod(ds)) / (_N ** _N)


def _soft_max_log(x, s=50.0):
    # smooth max of log(d^2) via log-sum-exp
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m))) / s


def _objective(x, lam):
    P = x.reshape(_N, 2)
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2)
    d2 = np.clip(d2, 1e-300, None)
    lp = 0.5 * np.sum(np.log(d2)[np.triu_indices(_N, k=1)])
    sm = _soft_max_log(np.log(d2), 50.0)
    return -(lp - lam * sm)


def _lgbf_polish(P, lam, maxiter=400):
    x0 = P.reshape(-1)
    res = minimize(_objective, x0, args=(lam,), method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 4000})
    return res.x.reshape(_N, 2)


def _slsqp_polish(P):
    x0 = P.reshape(-1)
    cons = [{"type": "ineq",
             "fun": lambda x: 4.0 - _diameter(x.reshape(_N, 2)) ** 2}]
    res = minimize(lambda x: -_log_prod(x.reshape(_N, 2)), x0, method="SLSQP",
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
    return res.x.reshape(_N, 2)


def _scale_diam2(P):
    dmax = _diameter(P)
    if dmax <= 0:
        return P
    return P * (2.0 / dmax)


def _regular_octagon():
    return np.array([[math.cos(2 * math.pi * k / _N),
                      math.sin(2 * math.pi * k / _N)] for k in range(_N)])


def _make_seeds(rng):
    seeds = []
    # regular octagon
    seeds.append(_regular_octagon().copy())
    # cage: regular with alternating small angular perturbations (breaks symmetry)
    base = _regular_octagon()
    for amp in (0.05, 0.1, 0.15, 0.2):
        ang = 2 * math.pi * np.arange(_N) / _N + amp * np.sin(2 * ang)
        seeds.append(np.column_stack([np.cos(ang), np.sin(ang)]))
    # random on unit disk
    for _ in range(4):
        r = np.sqrt(rng.random(_N))
        th = 2 * math.pi * rng.random(_N)
        seeds.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    # random near unit circle
    for _ in range(4):
        th = np.sort(2 * math.pi * rng.random(_N))
        r = 1.0 + 0.1 * rng.random(_N)
        seeds.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    # two-cluster "dumbbell" anti-symmetric seed
    th = 2 * math.pi * np.arange(_N) / _N + 0.3 * np.sin(4 * ang if False else 2 * (2 * math.pi * np.arange(_N) / _N))
    r = np.where(np.arange(_N) % 2 == 0, 1.0, 0.7)
    seeds.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    return seeds


def construct():
    rng = np.random.default_rng(12345)
    best_P = _regular_octagon()
    best_score = _score(best_P)

    def consider(P):
        nonlocal best_P, best_score
        P = np.asarray(P, dtype=float)
        if P.shape != (_N, 2) or not np.all(np.isfinite(P)):
            return
        # distinctness guard
        if _diameter(P) <= 0:
            return
        s = _score(P)
        if s > best_score:
            best_score = s
            best_P = P.copy()

    # ---- Stage 1: polish diverse seeds with L-BFGS-B (surrogate diameter) ----
    for P0 in _make_seeds(rng):
        if _elapsed() > _DEADLINE:
            break
        for lam in (0.5, 1.0, 2.0):
            P1 = _lgbf_polish(P0, lam)
            consider(P1)
            # exact SLSQP polish on the best
            P2 = _slsqp_polish(P1)
            consider(P2)
            if _elapsed() > _DEADLINE:
                break

    # ---- Stage 2: DE global search over full 16-dim space ----
    if _elapsed() < _DEADLINE - 25.0:
        bounds = [(-2.0, 2.0)] * (2 * _N)

        def de_obj(x):
            P = x.reshape(_N, 2)
            if _diameter(P) <= 0:
                return 1e9
            return -_log_prod(P)

        try:
            de = differential_evolution(de_obj, bounds, seed=7,
                                        maxiter=40, popsize=12,
                                        tol=1e-6, polish=False,
                                        init="sobol")
            for x in (de.x,):
                P = x.reshape(_N, 2)
                P = _lgbf_polish(P, 1.0)
                consider(P)
                P = _slsqp_polish(P)
                consider(P)
        except Exception:
            pass

    # ---- Stage 3: final tight polish on the global best ----
    if _elapsed() < _DEADLINE - 5.0:
        P = _slsqp_polish(best_P)
        consider(P)
        P = _lgbf_polish(P, 1.0, maxiter=800)
        consider(P)
        P = _slsqp_polish(P)
        consider(P)

    # ---- Final: scale to diameter 2 and return ----
    out = _scale_diam2(best_P)
    return [[float(out[i, 0]), float(out[i, 1])] for i in range(_N)]


if __name__ == "__main__":
    pts = construct()
    print("score:", _score(np.array(pts)))
    print(pts)
