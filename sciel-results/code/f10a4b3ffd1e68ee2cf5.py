# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 620.0  # safety margin before the 630 s evaluator kill


def _remaining():
    return _DEADLINE - time.time()


def _keich(N):
    """Keich binary construction: N base positions in [0,1), one per grid index."""
    pos = int(round(np.log2(N)))
    j_terms = np.arange(1, pos + 1)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        b = list(str(bin(idx))[2:])
        eps = np.array(b, dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        out[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return out


def _union_length(t, w, gl_x, gl_w):
    """Exact length of union of N unit-height intervals [t_i, t_i+w] via Gauss-Legendre."""
    N = len(t)
    lo = np.min(t)
    hi = np.max(t) + w
    if hi - lo <= 1e-9:
        return hi - lo
    m = len(gl_x)
    s = 0.0
    for i in range(N):
        a = t[i]
        b = t[i] + w
        if b <= lo or a >= hi:
            continue
        xl = np.maximum(a, lo)
        xr = np.minimum(b, hi)
        if xr - xl <= 0.0:
            continue
        c = 0.5 * (xl + xr)
        h = 0.5 * (xr - xl)
        z = gl_x + c
        s += np.sum(gl_w * (z >= a) * (z <= b))
    return (hi - lo) / m * s


def _logsumexp(v, beta):
    v = np.asarray(v, dtype=np.float64)
    m = np.max(v)
    return m + np.log(np.sum(np.exp(-beta * (v - m)))) / beta


def _soft_union(t, w, gl_x, gl_w, beta, eps):
    """Smooth (log-sum-exp) surrogate for the union length; strictly positive."""
    N = len(t)
    lo = np.min(t)
    hi = np.max(t) + w
    if hi - lo <= 1e-9:
        return hi - lo + eps
    m = len(gl_x)
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    z = gl_x + c
    v = np.zeros(m, dtype=np.float64)
    for i in range(N):
        a = t[i]
        b = t[i] + w
        if b <= lo or a >= hi:
            continue
        xl = np.maximum(a, lo)
        xr = np.minimum(b, hi)
        if xr - xl <= 0.0:
            continue
        ci = 0.5 * (xl + xr)
        hi2 = 0.5 * (xr - xl)
        zi = gl_x + ci
        v += hi2 * (zi >= a) * (zi <= b)
    v = v / m
    return _logsumexp(v, beta) + eps


def _vol_from_1d(Lx, Ly):
    return Lx * Ly


def _optimize_1d(seed, w, gl_x, gl_w, beta, eps, time_budget):
    """Diverse-restart coordinate descent on a single 1-D union-length problem."""
    N = int(round(1.0 / w))
    best = None
    best_val = np.inf
    n_restarts = 0
    while time.time() < time_budget and n_restarts < 40:
        n_restarts += 1
        r = np.random.default_rng(n_restarts * 101 + 7)
        if n_restarts == 1:
            x0 = _keich(N)
        elif n_restarts % 3 == 0:
            x0 = np.sort(r.uniform(0.0, 1.0, N))
        elif n_restarts % 3 == 1:
            x0 = np.sort(_keich(N) + 0.04 * (r.standard_normal(N)))
        else:
            x0 = np.sort(_keich(N) * r.uniform(0.7, 1.3) + 0.02 * r.standard_normal(N))
        x0 = np.clip(x0, -1.0, 2.0)
        res = minimize(
            lambda x: _soft_union(x, w, gl_x, gl_w, beta, eps),
            x0, method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9},
        )
        x = np.sort(res.x)
        val = _union_length(x, w, gl_x, gl_w)
        if val < best_val:
            best_val = val
            best = x
    if best is None:
        best = _keich(N)
        best_val = _union_length(best, w, gl_x, gl_w)
    return best, best_val


def _montecarlo_volume(positions, seed=12345, n=500000):
    x = positions[:, 0]
    y = positions[:, 1]
    N = 8
    i = np.arange(N)
    X0 = np.min(x)
    X1 = np.max(x + (i + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (i + 1) / N)
    if not (np.isfinite(X0) and np.isfinite(X1) and np.isfinite(Y0) and np.isfinite(Y1)):
        return None
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return None
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n)
    b = rng.uniform(Y0, Y1, n)
    c = rng.uniform(0.0, 1.0, n)
    inside = np.zeros(n, dtype=bool)
    for k in range(64):
        ii = k // N
        jj = k % N
        cx = x[k] + c * ii / N
        cy = y[k] + c * jj / N
        inside |= (a >= cx) & (a <= cx + 1.0 / N) & (b >= cy) & (b <= cy + 1.0 / N)
    frac = np.count_nonzero(inside) / n
    V = frac * (X1 - X0) * (Y1 - Y0)
    if not np.isfinite(V) or V <= 0.0:
        return None
    return V


def _validate(positions):
    """Return (valid, V)."""
    if positions.shape != (64, 2):
        return False, None
    if not np.all(np.isfinite(positions)):
        return False, None
    V = _montecarlo_volume(positions)
    if V is None:
        return False, None
    return True, V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    gl_x, gl_w = np.polynomial.legendre.leggauss(80)
    beta = 12.0
    eps = 1e-9

    # --- 1-D independent optimization (exploits the product structure) ---
    t1 = time.time()
    x1d, Lx = _optimize_1d("x", w, gl_x, gl_w, beta, eps, time_budget=_T0 + 300.0)
    y1d, Ly = _optimize_1d("y", w, gl_x, gl_w, beta, eps, time_budget=_T0 + 520.0)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    positions = np.stack([x1d[i_flat], y1d[j_flat]], axis=1).astype(np.float64)

    # --- Joint 128-D polish on the smooth 3-D surrogate (if time remains) ---
    if _remaining() > 40.0:
        def f3(p):
            X0 = np.min(p[:64])
            X1 = np.max(p[:64] + (i_flat + 1) / N)
            Y0 = np.min(p[64:])
            Y1 = np.max(p[64:] + (j_flat + 1) / N)
            if (X1 - X0) * (Y1 - Y0) <= 1e-9:
                return 1e6
            m = len(gl_x)
            c = 0.5 * (0.0 + 1.0)
            h = 0.5 * (1.0 - 0.0)
            zc = gl_x + c
            v = np.zeros(m, dtype=np.float64)
            for k in range(64):
                xx = p[k] + zc * i_flat[k] / N
                yy = p[64 + k] + zc * j_flat[k] / N
                v += (zc >= xx) * (zc <= xx + w) * (zc >= yy) * (zc <= yy + w)
            v = v / m
            return (X1 - X0) * (Y1 - Y0) * _logsumexp(v, beta) + eps

        p0 = positions.flatten().copy()
        res = minimize(f3, p0, method="L-BFGS-B",
                       options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-9,
                                "maxls": 20})
        if _remaining() > 20.0 and np.all(np.isfinite(res.x)):
            cand = res.x.reshape(64, 2)
            ok, Vc = _validate(cand)
            if ok:
                ok2, Vb = _validate(positions)
                if not ok2 or Vc < Vb:
                    positions = cand

    # --- Final safety check: fall back to seed if invalid ---
    ok, V = _validate(positions)
    if not ok:
        seed_pos = np.stack([_keich(N)[i_flat], _keich(N)[j_flat]], axis=1)
        positions = seed_pos

    if _remaining() <= 0.0:
        # deadline guard: return the best finite result found
        pass

    return positions


# EVOLVE-BLOCK-END
