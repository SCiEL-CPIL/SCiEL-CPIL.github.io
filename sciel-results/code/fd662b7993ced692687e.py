# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 10
_DEMANDS = (
    (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
    (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
    (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
)
_D = len(_DEMANDS)


def _build():
    cw = [np.zeros(_N, dtype=np.float64) for _ in range(_D)]
    ccw = [np.zeros(_N, dtype=np.float64) for _ in range(_D)]
    for i, (s, t, w) in enumerate(_DEMANDS):
        if t > s:
            for e in range(s, t):
                cw[i][e] = w
        else:
            for e in range(s, _N):
                cw[i][e] = w
            for e in range(0, t):
                cw[i][e] = w
        ccw[i] = w - cw[i]
    return cw, ccw


_CW, _CCW = _build()


def _maxload(cw, ccw, x):
    loads = np.zeros(_N, dtype=np.float64)
    for i in range(_D):
        if x[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return float(loads.max())


def _init(x, cw, ccw):
    loads = np.zeros(_N, dtype=np.float64)
    for i in range(_D):
        if x[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return float(loads.max())


def _init_delta(x, i, cw, ccw):
    """Return (new_loads, new_max) after flipping demand i from current state."""
    loads = np.zeros(_N, dtype=np.float64)
    for j in range(_D):
        if j == i:
            continue
        if x[j] >= 0.5:
            loads += ccw[j]
        else:
            loads += cw[j]
    if x[i] >= 0.5:
        loads += cw[i]
    else:
        loads += ccw[i]
    return float(loads.max())


def _hill(x, cw, ccw, max_iter, deadline):
    best = _init(x, cw, ccw)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        i = random.randrange(_D)
        x[i] = 1.0 - x[i]
        cand = _init(x, cw, ccw)
        if cand < best:
            best = cand
        else:
            x[i] = 1.0 - x[i]
    return x, best


def _hill_double(x, cw, ccw, max_iter, deadline):
    """Hill climbing with single and double flips."""
    best = _init(x, cw, ccw)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.7:
            # Single flip
            i = random.randrange(_D)
            x[i] = 1.0 - x[i]
            cand = _init(x, cw, ccw)
            if cand < best:
                best = cand
            else:
                x[i] = 1.0 - x[i]
        else:
            # Double flip
            i, j = random.sample(range(_D), 2)
            x[i] = 1.0 - x[i]
            x[j] = 1.0 - x[j]
            cand = _init(x, cw, ccw)
            if cand < best:
                best = cand
            else:
                x[i] = 1.0 - x[i]
                x[j] = 1.0 - x[j]
    return x, best


def _anneal(x, cw, ccw, t0, t1, iters, deadline):
    cur = _init(x, cw, ccw)
    best = cur
    bestx = x.copy()
    for it in range(iters):
        if time.time() > deadline:
            break
        # Adaptive temperature
        frac = it / max(iters, 1)
        t = t0 * (t1 / t0) ** frac
        i = random.randrange(_D)
        x[i] = 1.0 - x[i]
        cand = _init(x, cw, ccw)
        if cand < cur or random.random() < np.exp(-(cand - cur) / max(t, 1e-9)):
            cur = cand
            if cur < best:
                best = cur
                bestx = x.copy()
        else:
            x[i] = 1.0 - x[i]
    return bestx, best


def ring_routing() -> np.ndarray:
    random.seed(0)
    start = time.time()
    deadline = start + 600.0
    cw, ccw = _CW, _CCW

    # baseline: all clockwise
    x0 = np.zeros(_D)
    bestx, best = _hill(x0, cw, ccw, 20000, deadline)

    # multi-start hill climbing with double-flip neighborhood
    while time.time() < deadline - 5.0:
        x = (np.random.rand(_D) < 0.5).astype(np.float64)
        _, s = _hill_double(x, cw, ccw, 8000, deadline)
        if s < best:
            best = s
            bestx = x.copy()

    # simulated annealing for fine polishing
    while time.time() < deadline - 2.0:
        x = bestx.copy()
        _, s = _anneal(x, cw, ccw, 1.0, 0.01, 20000, deadline)
        if s < best:
            best = s
            bestx = x.copy()

    # final polish with double-flip neighborhood
    bestx, best = _hill_double(bestx, cw, ccw, 20000, deadline)
    return bestx


# EVOLVE-BLOCK-END
