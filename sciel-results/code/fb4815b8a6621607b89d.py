"""Fresh design for the 50-equal-circles-in-a-unit-circle packing task."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _hex_seed(r, jseed):
    """Hexagonal packing (spacing 2r) clipped to the disk, jittered by jseed.

    Returns an (N,2) array of centers, or None if fewer than N fit.
    """
    s = 2.0 * r
    R = 1.0
    K = int(np.ceil(R / s)) + 2
    h = s * np.sqrt(3.0) / 2.0
    pts = []
    for j in range(-K, K + 1):
        y = j * h
        if abs(y) > R:
            continue
        xoff = (s / 2.0) if (j % 2) else 0.0
        for i in range(-K, K + 1):
            x = i * s + xoff
            if x * x + y * y <= R * R:
                pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    if pts.shape[0] < N:
        return None
    # Prefer points near the center (stable packing core), then jitter.
    order = np.argsort(np.sum(pts * pts, axis=1), kind="stable")
    sel = pts[order[:N]]
    if jseed is not None:
        rng = np.random.default_rng(jseed)
        sel = sel + rng.uniform(-0.25, 0.25, size=sel.shape) * r
    return sel


def _score(P):
    """Raw packing radius (0.0 if any center is outside the unit disk)."""
    n = P.shape[0]
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    inside = 1.0 - np.sqrt(np.sum(P * P, axis=1)).max()
    r = min(min_pair, inside)
    if r < 0.0:
        return 0.0
    return float(r)


def _neg_smooth(P, t):
    """Negative of a smooth bottleneck approximation to the true radius."""
    n = P.shape[0]
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    inside = 1.0 - np.sqrt(np.sum(P * P, axis=1))
    vals = np.concatenate([pair.ravel(), inside.ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    m = vals.min()
    s = -np.log(np.sum(np.exp(-(vals - m) / t))) * t
    return -s


def _neg_smooth_grad(P, t):
    n = P.shape[0]
    d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    inside = 1.0 - np.sqrt(np.sum(P * P, axis=1))
    vals = np.concatenate([pair.ravel(), inside.ravel()])
    vals = vals[np.isfinite(vals)]
    m = vals.min()
    w = np.exp(-(vals - m) / t)
    ssum = np.sum(w)
    grad = np.zeros_like(P)
    # pair part: vals = d/2, d = |P_i-P_j|, d/dP_i = (P_i-P_j)/d
    pw = w[:n * n].reshape(n, n)
    diff = P[:, None, :] - P[None, :, :]
    invd = np.zeros_like(d)
    mask = d > 0
    invd[mask] = 1.0 / d[mask]
    # contribution to grad[i] from all j: (1/2) * (P_i - P_j)/d_ij * w_ij
    gpair = np.sum((diff * (pw * invd / 2.0)[:, :, None]), axis=1)
    # inside part: vals = 1 - |P_i|, d/dP_i = -P_i/|P_i|
    iw = w[n * n:]
    rad = np.sqrt(np.sum(P * P, axis=1))
    rad_safe = np.where(rad > 1e-12, rad, 1e-12)
    ginside = -iw[:, None] * (P / rad_safe[:, None])
    grad = gpair + ginside
    # scale: d s/dP = -sum w * (d vals/dP) / ssum  ... s = -t log sum exp(-(v-m)/t)
    # ds/dP_i = -t * (1/ssum) * sum_j w_j * (d v_j/dP_i) * (-1/t) = (1/ssum) sum_j w_j d v_j/dP_i
    # we computed d v/dP (v=pair/inside). neg_smooth = -s, so grad = -ds/dP = -(1/ssum) sum w dv/dP
    grad = -grad / ssum
    return grad


def _optimize(P0, t, time_budget):
    """Run L-BFGS-B then Nelder-Mead on a smooth objective; return best P."""
    best = P0.copy()
    best_r = _score(P0)
    try:
        res = minimize(
            _neg_smooth, P0, jac=_neg_smooth_grad, method="L-BFGS-B",
            args=(t,),
            options={"maxiter": 60, "gtol": 1e-5, "maxfun": 400},
        )
        P1 = np.asarray(res.x, dtype=np.float64)
        r1 = _score(P1)
        if r1 > best_r:
            best, best_r = P1, r1
    except Exception:
        pass
    # Nelder-Mead refinement (no gradient, robust to the kink)
    try:
        res2 = minimize(
            _neg_smooth, best, method="Nelder-Mead", args=(t,),
            options={"maxiter": 40, "xatol": 1e-7, "fatol": 1e-9},
        )
        P2 = np.asarray(res2.x, dtype=np.float64)
        r2 = _score(P2)
        if r2 > best_r:
            best, best_r = P2, r2
    except Exception:
        pass
    return best


def construct():
    import time
    deadline = time.time() + 135.0  # safety margin below 150 s
    t = 0.004

    # Initial radius estimate from the hex lattice.
    base_r = 0.125
    P_best = None
    r_best = 0.0

    # Build a clean hex seed at the base radius.
    P0 = _hex_seed(base_r, None)
    if P0 is not None:
        r0 = _score(P0)
        if r0 > r_best:
            r_best, P_best = r0, P0

    # Jittered restarts + optimization, keeping the best.
    jitters = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    for js in jitters:
        if time.time() > deadline:
            break
        P = _hex_seed(base_r, js)
        if P is None:
            continue
        cand = _optimize(P, t, deadline)
        rc = _score(cand)
        if rc > r_best:
            r_best, P_best = rc, cand

    # One more pass at a slightly larger radius to test feasibility headroom.
    if time.time() < deadline - 25.0:
        for js in [None, 0, 1]:
            if time.time() > deadline:
                break
            P = _hex_seed(base_r * 1.02, js)
            if P is None:
                continue
            cand = _optimize(P, t, deadline)
            rc = _score(cand)
            if rc > r_best:
                r_best, P_best = rc, cand

    if P_best is None:
        # Fallback: simple ring packing (always valid).
        P_best = np.zeros((N, 2))
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        P_best[:, 0] = 0.6 * np.cos(ang)
        P_best[:, 1] = 0.6 * np.sin(ang)
    return P_best
