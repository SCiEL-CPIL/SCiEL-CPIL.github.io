# EVOLVE-BLOCK-START
import numpy as np
import time

def _setup(n=64, M=20000):
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    perm = np.argsort(a)          # order of a (== order of L at x=0)
    lo = perm[:-1]
    hi = perm[1:]
    return y, a, w, lo, hi

def _area_fast(x, y, a, w, lo, hi, n=64, M=20000):
    """Area via fixed consecutive-pair ordering. O(M)."""
    Llo = x[lo][:, np.newaxis] + a[lo][:, np.newaxis] * y[np.newaxis, :]
    Lhi = x[hi][:, np.newaxis] + a[hi][:, np.newaxis] * y[np.newaxis, :]
    d = Lhi - Llo
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def _area_ref(x, n=64, M=20000):
    """Correct area via full sort at each height. O(M n log n)."""
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    Ls = np.sort(L, axis=0)
    d = Ls[1:, :] - Ls[:-1, :]
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def _order_ok(x, y, a, lo, hi, n=64):
    """Check whether the fixed a-ordering equals the true L-ordering at all heights."""
    Llo = x[lo][:, np.newaxis] + a[lo][:, np.newaxis] * y[np.newaxis, :]
    Lhi = x[hi][:, np.newaxis] + a[hi][:, np.newaxis] * y[np.newaxis, :]
    return bool(np.all(Lhi - Llo >= -1e-12))

def _area_grad(x, y, a, w, lo, hi, n=64, M=20000):
    """Area + subgradient under fixed ordering. O(M)."""
    Llo = x[lo][:, np.newaxis] + a[lo][:, np.newaxis] * y[np.newaxis, :]
    Lhi = x[hi][:, np.newaxis] + a[hi][:, np.newaxis] * y[np.newaxis, :]
    d = Lhi - Llo
    sat = (d < w[np.newaxis, :])
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    area = float(np.mean(union))
    g = np.zeros(n, dtype=np.float64)
    g[hi] += np.sum(sat, axis=1) / M
    g[lo] -= np.sum(sat, axis=1) / M
    return area, g

def _subgrad(x0, y, a, w, lo, hi, n, M, deadline):
    x = x0.copy()
    area, g = _area_grad(x, y, a, w, lo, hi, n, M)
    v = np.zeros(n, dtype=np.float64)
    mu = 0.9
    step = 0.05
    while time.time() < deadline:
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        step_polyak = max(np.dot(x, g), 1e-12) / (gnorm * gnorm + 1e-12)
        step = min(step_polyak, 0.3)
        v = mu * v - step * g
        xn = x + v
        an, gn = _area_grad(xn, y, a, w, lo, hi, n, M)
        if an < area:
            x, area, g = xn, an, gn
        else:
            step *= 0.7
            v *= 0.5
            if step < 1e-12:
                break
    return x, area

def _local_search(x, y, a, w, lo, hi, n, M, deadline):
    """Vectorized greedy coordinate/swap local search (O(M) per trial)."""
    best = x.copy()
    best_a = _area_fast(best, y, a, w, lo, hi, n, M)
    rng = np.random.default_rng(7)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline:
                break
            for j in order:
                if j <= i or time.time() > deadline:
                    continue
                trial = best.copy()
                trial[i], trial[j] = best[j], best[i]
                ta = _area_fast(trial, y, a, w, lo, hi, n, M)
                if ta < best_a - 1e-15:
                    best = trial
                    best_a = ta
                    improved = True
    return best, best_a

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start = time.time()
    deadline = start + 600.0

    y, a, w, lo, hi = _setup(n, M)

    def evaluate(x):
        """Return (area, ordering_ok)."""
        if _order_ok(x, y, a, lo, hi, n):
            return _area_fast(x, y, a, w, lo, hi, n, M), True
        return _area_ref(x, n, M), False

    # Seed candidates (analytic structures)
    rng = np.random.default_rng(12345)
    seeds = [
        np.zeros(n, dtype=np.float64),
        -a * 0.5,
        -a * (1.0 / 3.0),
        -a * (2.0 / 3.0),
        a * 0.5,
        -a * 0.5 + rng.normal(0, 0.3, n),
        rng.normal(0, 0.5, n),
    ]
    best_x, best_area = None, np.inf
    for s in seeds:
        ar, _ok = evaluate(s)
        if ar < best_area:
            best_area, best_x = ar, s.copy()

    # Phase 1: subgradient descent from a few restarts
    n_restarts = 6
    budget = deadline - start
    per = max(1.0, budget / n_restarts - 5.0)
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        x0 = np.zeros(n, dtype=np.float64) if r == 0 else rng.normal(0, 0.5, n)
        rd = min(deadline - 20.0, time.time() + per)
        xr, ar = _subgrad(x0, y, a, w, lo, hi, n, M, rd)
        ar, _ok = evaluate(xr)
        if ar < best_area:
            best_area, best_x = ar, xr.copy()

    # Phase 2: local-search polish
    if time.time() < deadline - 20:
        best_x, best_area = _local_search(best_x, y, a, w, lo, hi, n, M,
                                          deadline=deadline - 15)

    # Phase 3: short subgradient + polish from perturbed point
    if time.time() < deadline - 15:
        xr, ar = _subgrad(best_x + rng.normal(0, 0.1, n), y, a, w, lo, hi,
                          n, M, min(deadline - 15, time.time() + 8.0))
        ar, _ok = evaluate(xr)
        if ar < best_area:
            best_area, best_x = ar, xr.copy()
            if time.time() < deadline - 10:
                best_x, best_area = _local_search(best_x, y, a, w, lo, hi, n, M,
                                                  deadline=deadline - 10)

    # Final validation
    if not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    ref = _area_ref(best_x, n, M)
    if not np.isfinite(ref) or ref <= 0:
        return np.zeros(n, dtype=np.float64)
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
