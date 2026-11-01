"""Circle packing: maximize common radius of 50 equal circles in the unit disk.

Uses a smooth (log-sum-exp) surrogate for the min objective plus a soft
penalty for overlaps / container violations, optimized with L-BFGS-B from
multiple diverse starts under a hard wall-clock deadline.
"""
import time
import numpy as np

N = 50

_DEADLINE = 150.0
_START = time.time()
_TIME_BUDGET = 110.0  # safety margin before the 150s kill

# ---- smooth objective ----------------------------------------------------
_EPS = 1e-9


def _gaps(P):
    """Pairwise gap matrix and container margins.

    gap_ij = |c_i - c_j| - 2r   (want >= 0)
    margin_i = 1 - |c_i| - r     (want >= 0)
    """
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    gap = d - 2.0 * r
    margin = 1.0 - np.sqrt(np.sum(P * P, axis=1)) - r
    return gap, margin


def _softmin(vals, t):
    """Smooth minimum of vals (1-D array)."""
    m = np.min(vals)
    return m + (1.0 / t) * np.log(np.sum(np.exp(t * (vals - m))))


def _objective(x):
    P = x.reshape(-1, 2)
    gap, margin = _gaps(P)
    vals = np.concatenate([gap.ravel(), margin])
    r = _softmin(vals, T)

    # penalties: overlap (gap<0) and outside (margin<0), smooth
    over = np.maximum(-gap.ravel(), 0.0)
    out = np.maximum(-margin, 0.0)
    pen = np.sum(over * over) + np.sum(out * out)
    return r - LAMBDA * pen


def _score(P):
    """True (raw) score: max common radius, or -inf if invalid."""
    P = np.asarray(P, dtype=float)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(P * P, axis=1))))
    return r


# ---- initial layouts -----------------------------------------------------
def _hex_init(spacing):
    pts = []
    k = int(1.0 / spacing) + 2
    s3 = spacing * np.sqrt(3.0) / 2.0
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * s3
            pts.append((x, y))
    P = np.array(pts)
    rad = np.hypot(P[:, 0], P[:, 1])
    P = P[rad <= 1.0 - spacing / 2.0]
    P = P[np.argsort(rad, kind="stable")]
    return P[:N]


def _ring_init(rings, counts, radii):
    pts = []
    for rad, cnt in zip(radii, counts):
        if cnt == 1:
            pts.append((0.0, 0.0))
            continue
        th = 2.0 * np.pi * np.arange(cnt) / cnt
        pts.extend((rad * np.cos(t), rad * np.sin(t)) for t in th)
    P = np.array(pts)
    if len(P) < N:
        P = np.vstack([P, np.zeros((N - len(P), 2))])
    return P[:N]


def _random_init(rng):
    P = rng.uniform(-0.85, 0.85, size=(N, 2))
    # keep inside with margin
    rad = np.hypot(P[:, 0], P[:, 1])
    P = P * (0.85 / np.maximum(rad, 1e-9))[:, None]
    return P


def _starts():
    starts = []
    # hex lattice
    starts.append(_hex_init(0.26))
    starts.append(_hex_init(0.28))
    # ring layouts (concentric), several variants
    starts.append(_ring_init(None, [1, 6, 12, 18, 13], [0.0, 0.30, 0.58, 0.82, 0.98]))
    starts.append(_ring_init(None, [1, 6, 12, 18, 13], [0.0, 0.32, 0.60, 0.84, 0.99]))
    starts.append(_ring_init(None, [1, 7, 13, 18, 11], [0.0, 0.30, 0.58, 0.83, 0.98]))
    starts.append(_ring_init(None, [1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.97]))
    # random starts
    rng = np.random.default_rng(12345)
    for s in range(4):
        starts.append(_random_init(rng))
    return starts


def _optimize(P0, max_iter, time_left_fn):
    x0 = P0.reshape(-1).copy()
    t0 = time.time()
    res = None
    try:
        from scipy.optimize import minimize
        res = minimize(
            _objective, x0, method="L-BFGS-B",
            options={"maxiter": max_iter, "maxfun": max_iter * 3,
                     "maxls": 40, "maxcor": 12},
            callback=None,
        )
    except Exception:
        return None
    return res.x.reshape(-1, 2)


def construct():
    global T, LAMBDA
    rng = np.random.default_rng(7)
    starts = _starts()

    best = None
    best_score = -np.inf

    # anneal the temperature: start coarse, end fine
    T_sched = [30.0, 60.0, 120.0, 240.0, 480.0]
    LAMBDA = 2.0

    for Ti in T_sched:
        T = Ti
        if time.time() - _START > _TIME_BUDGET:
            break
        for P0 in starts:
            if time.time() - _START > _TIME_BUDGET:
                break
            P = _optimize(P0, max_iter=300, time_left_fn=None)
            if P is None:
                continue
            sc = _score(P)
            if np.isfinite(sc) and sc > best_score:
                best_score = sc
                best = P.copy()
            # also keep the raw optimizer output even if slightly worse
            if np.isfinite(sc) and sc > best_score - 1e-6 and best is None:
                best = P.copy()

    # final high-precision polish on the best point
    if best is not None and time.time() - _START < _TIME_BUDGET:
        T = 2000.0
        P = _optimize(best, max_iter=500, time_left_fn=None)
        if P is not None:
            sc = _score(P)
            if np.isfinite(sc) and sc > best_score:
                best = P
                best_score = sc

    # fallback: a guaranteed-valid hex packing
    if best is None or not np.all(np.isfinite(best)):
        P = _hex_init(0.28)
        if len(P) < N:
            P = _hex_init(0.30)
        best = P

    # ensure valid shape and finiteness
    best = np.asarray(best, dtype=float)
    if best.shape != (N, 2):
        best = _hex_init(0.30)
    if not np.all(np.isfinite(best)):
        best = _hex_init(0.30)

    return best
