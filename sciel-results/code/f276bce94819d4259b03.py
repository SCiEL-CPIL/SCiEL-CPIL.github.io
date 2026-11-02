import numpy as np
import time

N = 50

def _autoconv(h):
    n = len(h)
    c = np.zeros(2 * n - 1)
    for i in range(n):
        hi = h[i]
        if hi != 0.0:
            c[i:i + n] += hi * h
    return c

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    l2sq = float(h @ h)
    l1 = float(h.sum())
    if l1 < 1e-15:
        return 0.0
    m = float(_autoconv(h).max())
    if m < 1e-15:
        return 0.0
    return l2sq / (l1 * m)

def construct():
    t0 = time.time()
    dl = t0 + 120.0
    rng = np.random.default_rng(7)

    best = np.ones(N)
    best_r = _score(best)

    # Random sparse starts
    for _ in range(3000):
        if time.time() > dl - 15:
            break
        k = int(rng.integers(3, 21))
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = rng.random(k)
        r = _score(h)
        if r > best_r:
            best_r, best = r, h.copy()

    # Coordinate polish
    h = best.copy()
    for _ in range(80000):
        if time.time() > dl - 3:
            break
        j = int(rng.integers(N))
        o = h[j]
        h[j] = max(0.0, o + rng.normal(0, 0.15))
        r = _score(h)
        if r > best_r:
            best_r, best = r, h.copy()
        else:
            h[j] = o

    return best.tolist()
