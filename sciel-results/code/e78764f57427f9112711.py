# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 10
_DEMANDS = [
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
]
_D = len(_DEMANDS)
_CCW_MASK = np.zeros((_D, _N), dtype=np.int64)


def _build_masks():
    for i, (s, t, w) in enumerate(_DEMANDS):
        if s == t:
            continue
        for e in range(s, t):          # clockwise edges s..t-1
            _CCW_MASK[i, (e + 1) % _N] = 1   # the *other* edges = counter-clockwise


_build_masks()
_W = np.array([w for (_, _, w) in _DEMANDS], dtype=np.float64)
_WS = _W.sum()


def _loads(routing):
    # routing[i] >= 0.5 => counter-clockwise => add weight to CCW edges
    return _CCW_MASK[routing >= 0.5].astype(np.float64) @ _W


def _bb_optimum(deadline):
    """Exact branch-and-bound minimizing max edge load."""
    order = list(range(_D))
    order.sort(key=lambda i: _W[i], reverse=True)
    best = {"ml": _WS, "routing": np.zeros(_D, dtype=np.float64)}
    ml = np.zeros(_N, dtype=np.float64)
    routing = np.zeros(_D, dtype=np.float64)
    rem = np.empty(_D + 1, dtype=np.float64)
    rem[_D] = 0.0
    for k in range(_D - 1, -1, -1):
        rem[k] = rem[k + 1] + _W[order[k]]

    def rec(k):
        if time.time() > deadline:
            return
        d = order[k]
        if k == _D:
            c = ml.max()
            if c < best["ml"] - 1e-9:
                best["ml"] = c
                best["routing"] = routing.copy()
            return
        for ccw in (0, 1):
            if ml.max() + rem[k + 1] >= best["ml"] - 1e-9:
                break
            if ccw:
                routing[d] = 1.0
                ml += _W[d] * _CCW_MASK[d]
            else:
                routing[d] = 0.0
            rec(k + 1)
            if ccw:
                ml -= _W[d] * _CCW_MASK[d]

    rec(0)
    return best["routing"], best["ml"]


def _local_search(routing, deadline):
    """Single-flip local search minimizing max edge load (only accepts strict improvement)."""
    ml = _loads(routing)
    cur = ml.max()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(_D):
            if routing[i] >= 0.5:
                delta = -_W[i] * _CCW_MASK[i]
            else:
                delta = _W[i] * _CCW_MASK[i]
            cand = ml + delta
            if cand.max() < cur - 1e-9:
                routing[i] = 1.0 - routing[i]
                ml = cand
                cur = ml.max()
                improved = True
    return routing, cur


def ring_routing() -> np.ndarray:
    deadline = time.time() + 590.0   # explicit safety margin vs 600s limit

    # Core: exact optimum via branch-and-bound (essentially instant for 15 demands).
    best_r, best_ml = _bb_optimum(deadline)

    # Safety net: time-boxed metaheuristic that can only improve on the incumbent.
    try:
        rng = random.Random(12345)
        while time.time() < deadline:
            r = np.array([rng.random() for _ in range(_D)])
            r, ml = _local_search(r.copy(), deadline)
            if ml < best_ml - 1e-9:
                best_ml = ml
                best_r = r.copy()
    except Exception:
        pass

    return best_r


# EVOLVE-BLOCK-END
