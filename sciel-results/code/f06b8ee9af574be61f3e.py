"""
Heuristic for ACO on multiple-constraint knapsack.
Uses a fast proxy ACO to tune a small set of heuristic parameters.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 330.0
_SAFETY = 30.0  # stop new work this far before the kill


def _compute_heuristics(prizes, weights, wp, we, S, b):
    """
    heuristic = S * (wp * P + we * E) + b
    P = mean prize across constraints
    E = P / (mean weight + eps)
    """
    P = prizes.mean(axis=0)
    W = weights.mean(axis=0)
    E = P / (W + 1e-9)
    h = S * (wp * P + we * E) + b
    h = np.where(h > 0, h, 1e-9)
    return h


def _aco_run(prizes, weights, heuristic, n_ants=20, n_iter=30, cap=25.0, seed=0):
    """
    Fast ACO simulation matching the contract exactly. Returns best prize.
    """
    rng = np.random.RandomState(seed)
    n_items = prizes.shape[1]
    n_con = prizes.shape[0]

    pher = np.ones(n_items)
    best = 0.0

    for _ in range(n_iter):
        iter_best = 0.0
        for _ in range(n_ants):
            remaining = np.full(n_con, cap)
            total = 0.0

            # Precompute feasibility masks incrementally? No, just loop.
            # But we can vectorize the feasibility check.
            for _ in range(n_items):
                # Items that still fit in every constraint
                ok = np.all(weights <= remaining[:, None] + 1e-12, axis=0)
                if not np.any(ok):
                    break

                p = pher * heuristic
                p = np.where(ok, p, 0.0)
                s = p.sum()
                if s <= 0:
                    break

                idx = rng.choice(n_items, p=p / s)

                # Commit
                remaining -= weights[:, idx]
                total += prizes[:, idx].sum()

            if total > iter_best:
                iter_best = total

        if iter_best > best:
            best = iter_best

        # Pheromone update
        pher = pher * 0.9 + iter_best / 10.0

    return best


def _make_instance(seed):
    rng = np.random.RandomState(seed)
    prizes = rng.uniform(0, 1, size=(5, 100))
    weights = rng.uniform(0, 1, size=(5, 100))
    return prizes, weights


def _proxy_score(params, n_inst=5, n_ants=20, n_iter=30):
    wp, we, S, b = params
    total = 0.0
    for s in range(n_inst):
        prizes, weights = _make_instance(1000 + s)
        h = _compute_heuristics(prizes, weights, wp, we, S, b)
        total += _aco_run(prizes, weights, h, n_ants=n_ants, n_iter=n_iter, seed=1000 + s)
    return total / n_inst


def _optimize():
    """
    Tune heuristic parameters using Nelder-Mead on a proxy ACO.
    """
    from scipy.optimize import minimize

    best_params = np.array([1.0, 3.0, 50.0, 1.0])
    best_val = -1.0

    # A few random starts
    rng = np.random.RandomState(42)
    starts = [
        np.array([1.0, 3.0, 50.0, 1.0]),
        np.array([0.5, 5.0, 100.0, 0.5]),
        np.array([2.0, 1.0, 200.0, 2.0]),
        np.array([0.0, 6.0, 150.0, 0.0]),
        np.array([1.0, 10.0, 80.0, 0.5]),
        np.array([0.2, 8.0, 120.0, 0.1]),
    ]

    for start in starts:
        if time.time() > _DEADLINE - _SAFETY:
            break

        def obj(x):
            return -_proxy_score(x, n_inst=3, n_ants=10, n_iter=15)

        try:
            res = minimize(obj, start, method="Nelder-Mead",
                           options={"maxiter": 40, "xatol": 1e-3, "fatol": 1e-3})
            if res.fun < best_val:
                best_val = res.fun
                best_params = np.clip(res.x, 1e-6, None)
        except Exception:
            pass

        if time.time() > _DEADLINE - _SAFETY:
            break

    return best_params


def heuristics(prizes, weights):
    t0 = time.time()
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if prizes.shape != weights.shape or prizes.ndim != 2:
            n = prizes.shape[1] if prizes.ndim == 2 else 100
            return np.ones(n)

        if time.time() > _DEADLINE - _SAFETY:
            # Out of budget: fall back to a strong static model heuristic.
            wp, we, S, b = 1.0, 4.0, 100.0, 1.0
            return _compute_heuristics(prizes, weights, wp, we, S, b)

        params = _optimize()
        h = _compute_heuristics(prizes, weights, *params)

        if not np.all(np.isfinite(h)):
            h = _compute_heuristics(prizes, weights, 1.0, 4.0, 100.0, 1.0)

        # Guard: every entry positive.
        h = np.where(h > 0, h, 1e-9)
        return h

    except Exception:
        # Last-resort safe fallback
        try:
            P = np.asarray(prizes, dtype=float).mean(axis=0)
            W = np.asarray(weights, dtype=float).mean(axis=0)
            h = P / (W + 1e-9) + 0.5
            h = np.where(h > 0, h, 1e-9)
            return h
        except Exception:
            return np.ones(100)
# EVOLVE-BLOCK-END
