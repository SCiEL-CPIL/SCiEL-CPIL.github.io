# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, line_search

_START = time.perf_counter()
_DEADLINE = _START + 540.0  # hard safety margin under 600 s limit


def _time_left():
    return _DEADLINE - time.perf_counter()


def _exact_1d_objective(p):
    """Exact integral over c in [0,1] of (union length of N sheared unit-intervals)^2.
    p = base x positions (one per grid index i). Interval i at height c:
        [p_i + c*i/N, p_i + 1/N + c*i/N].  All N intervals have the same width 1/N.
    The union length L(c) is piecewise-linear in c; breakpoints are the c-values where
    any interval endpoint equals another's. We integrate L(c)^2 exactly per segment.
    """
    N = 8
    w = 1.0 / N
    p = np.asarray(p, dtype=np.float64)
    i = np.arange(N, dtype=np.float64)

    # endpoints as affine functions of c: e = a + b*c
    a_lo = p
    b_lo = i / N
    a_hi = p + w
    b_hi = i / N
    A = np.concatenate([a_lo, a_hi])
    B = np.concatenate([b_lo, b_hi])
    # breakpoints: c where A[m] + B[m]*c == A[n] + B[n]*c
    cbr = [0.0, 1.0]
    for m in range(2 * N):
        for n in range(2 * N):
            if m == n:
                continue
            db = B[m] - B[n]
            if abs(db) < 1e-15:
                continue
            c = (A[n] - A[m]) / db
            if -1e-12 < c < 1.0 + 1e-12:
                cbr.append(min(max(c, 0.0), 1.0))
    cbr = np.array(cbr, dtype=np.float64)
    cbr = np.unique(np.round(cbr, 14))
    order = np.argsort(cbr)
    cbr = cbr[order]
    total = 0.0
    for k in range(len(cbr) - 1):
        c0, c1 = cbr[k], cbr[k + 1]
        if c1 - c0 <= 1e-15:
            continue
        # L(c) = max e(c) - min e(c); e are affine. L is linear on [c0,c1].
        # sample interior point
        cm = 0.5 * (c0 + c1)
        e = A + B * cm
        Lm = e.max() - e.min()
        e0 = A + B * c0
        L0 = e0.max() - e0.min()
        e1 = A + B * c1
        L1 = e1.max() - e1.min()
        # integral of L^2 over [c0,c1], L linear: (h/3)*(L0^2 + L0*L1 + L1^2)
        h = c1 - c0
        total += (h / 3.0) * (L0 * L0 + L0 * L1 + L1 * L1)
    return total


def _build_positions(px, py):
    """Return (64,2) array: row k -> tube (i,j), i=k//8, j=k%8. base x depends only on i,
    base y only on j (separable construction)."""
    N = 8
    i = np.repeat(np.arange(N), N)
    j = np.tile(np.arange(N), N)
    return np.stack([px[i], py[j]], axis=1)


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _polish_1d(x, obj, iters=60):
    """Coordinate descent with exact line search (1D convex per coordinate)."""
    x = np.array(x, dtype=np.float64, copy=True)
    N = len(x)
    for it in range(iters):
        if _time_left() < 15.0:
            break
        improved = False
        for k in range(N):
            f0 = obj(x)
            # 1D scan around current
            lo, hi = x[k] - 1.0, x[k] + 1.0
            # golden / parabolic refinement
            best_x = x[k]
            best_f = f0
            for _ in range(20):
                t = 0.5 * (lo + hi)
                x[k] = t
                f = obj(x)
                if f < best_f:
                    best_f = f
                    best_x = t
                # decide bracket
                x[k] = x[k]  # keep
                # simple: evaluate midpoint and compare with lo,hi endpoints via stored
                # (we just do a fine grid + parabolic)
                break
            # fine grid
            grid = np.linspace(lo, hi, 80)
            fs = np.empty(len(grid))
            for gi, gv in enumerate(grid):
                x[k] = gv
                fs[gi] = obj(x)
            gi = int(np.argmin(fs))
            if fs[gi] < best_f:
                best_f = fs[gi]
                best_x = grid[gi]
            # parabolic refinement
            if 0 < gi < len(grid) - 1:
                x1, x2, x3 = grid[gi - 1], grid[gi], grid[gi + 1]
                f1, f2, f3 = fs[gi - 1], fs[gi], fs[gi + 1]
                d = 0.5 * ((x1 - x3) * (f1 - f3) + (x1 - x2) * (f1 - f2) - (x3 - x2) * (f3 - f2))
                if abs(d) > 1e-15:
                    xp = x2 + d
                    if x1 <= xp <= x3:
                        x[k] = xp
                        fp = obj(x)
                        if fp < best_f:
                            best_f = fp
                            best_x = xp
            x[k] = best_x
            if best_x != x[k]:
                improved = True
        if not improved:
            break
    return x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich(N)

    # ---- solve 1D problem for x (base positions per grid index i) ----
    def obj_x(p):
        return _exact_1d_objective(p)

    best_x = np.array(keich, dtype=np.float64)
    best_xf = obj_x(best_x)

    # L-BFGS-B from several starts
    starts = [
        np.array(keich, dtype=np.float64),
        -np.array(keich, dtype=np.float64),
        np.zeros(N),
        np.linspace(-0.5, 0.5, N),
    ]
    for s in starts:
        if _time_left() < 60.0:
            break
        try:
            res = minimize(obj_x, s, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 4000, "gtol": 1e-9})
            if res.success or res.fun < best_xf:
                best_x = np.asarray(res.x, dtype=np.float64)
                best_xf = float(res.fun)
        except Exception:
            pass

    # polish
    best_x = _polish_1d(best_x, obj_x)
    best_xf = obj_x(best_x)

    # ---- solve 1D problem for y (independent, same objective) ----
    def obj_y(p):
        return _exact_1d_objective(p)

    best_y = np.array(keich, dtype=np.float64)
    best_yf = obj_y(best_y)
    for s in starts:
        if _time_left() < 60.0:
            break
        try:
            res = minimize(obj_y, s, method="L-BFGS-B",
                           options={"maxiter": 400, "maxfun": 4000, "gtol": 1e-9})
            if res.success or res.fun < best_yf:
                best_y = np.asarray(res.x, dtype=np.float64)
                best_yf = float(res.fun)
        except Exception:
            pass
    best_y = _polish_1d(best_y, obj_y)
    best_yf = obj_y(best_y)

    positions = _build_positions(best_x, best_y)

    # ---- validate & fallback ----
    if not np.all(np.isfinite(positions)):
        positions = _build_positions(keich, keich)

    # compute the evaluator's box to ensure area > 1e-9
    i = np.repeat(np.arange(N), N)
    j = np.tile(np.arange(N), N)
    X0 = float(positions[:, 0].min())
    X1 = float((positions[:, 0] + (i + 1) / N).max())
    Y0 = float(positions[:, 1].min())
    Y1 = float((positions[:, 1] + (j + 1) / N).max())
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        positions = _build_positions(keich, keich)

    return positions


# EVOLVE-BLOCK-END
