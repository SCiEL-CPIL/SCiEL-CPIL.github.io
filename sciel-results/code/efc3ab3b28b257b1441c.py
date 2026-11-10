# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 8
DEADLINE = time.time() + 600.0
MARGIN = 120.0

def _vol1d(X, n=N):
    # volume of union over c in [0,1] of intervals [X[i]+c*i/n, X[i]+c*i/n+1/n]
    n = int(n)
    I = np.arange(n, dtype=np.float64)
    lo = X[:, None] + (np.arange(n) / n)[None, :] * I[None, :]
    hi = lo + 1.0 / n
    # 1-D union measure of { (c,u): lo[i,c] <= u <= hi[i,c] }
    c = np.arange(401, dtype=np.float64) / 400.0
    L = np.maximum.reduce([np.full(401, -1e18), np.maximum.reduce(lo[:, c])])
    H = np.minimum.reduce([np.full(401, 1e18), np.minimum.reduce(hi[:, c])])
    m = np.maximum(0.0, H - L)
    return float(np.trapezoid(m, c))

def _vol2d(P, n=N, M=140):
    # union volume of 64 sheared tubes; P shape (64,2) base positions
    n = int(n)
    I = np.arange(n, dtype=np.float64)
    J = np.arange(n, dtype=np.float64)
    i = np.repeat(I, n)
    j = np.tile(J, n)
    X = P[:, 0]
    Y = P[:, 1]
    Xlo = X[:, None] + (i / n)[:, None] * np.arange(n)[None, :]
    Xhi = Xlo + 1.0 / n
    Ylo = Y[:, None] + (j / n)[:, None] * np.arange(n)[None, :]
    Yhi = Ylo + 1.0 / n
    c = np.arange(M, dtype=np.float64) / (M - 1)
    Lx = np.maximum.reduce([np.full(M, -1e18), np.maximum.reduce(Xlo[:, c])])
    Hx = np.minimum.reduce([np.full(M, 1e18), np.minimum.reduce(Xhi[:, c])])
    Ly = np.maximum.reduce([np.full(M, -1e18), np.maximum.reduce(Ylo[:, c])])
    Hy = np.minimum.reduce([np.full(M, 1e18), np.minimum.reduce(Yhi[:, c])])
    A = np.maximum(0.0, Hx - Lx) * np.maximum(0.0, Hy - Ly)
    return float(np.trapezoid(A, c))

def _vol_full(P, n=N):
    # fixed-sample MC estimate exactly as the evaluator (500k points)
    n = int(n)
    i = np.repeat(np.arange(n), n)
    j = np.tile(np.arange(n), n)
    X = P[:, 0]
    Y = P[:, 1]
    X0 = float(np.min(X))
    X1 = float(np.max(X + (i + 1) / n))
    Y0 = float(np.min(Y))
    Y1 = float(np.max(Y + (j + 1) / n))
    if not (np.isfinite(X0) and np.isfinite(X1) and np.isfinite(Y0) and np.isfinite(Y1)):
        return np.inf
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return np.inf
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    cc = rng.uniform(0.0, 1.0, 500000)
    inside = np.zeros(500000, dtype=bool)
    for k in range(64):
        xlo = X[k] + cc * i[k] / n
        ylo = Y[k] + cc * j[k] / n
        inside |= (a >= xlo) & (a <= xlo + 1.0 / n) & (b >= ylo) & (b <= ylo + 1.0 / n)
    V = inside.mean() * (X1 - X0) * (Y1 - Y0)
    return float(V)

def _keich(n=N):
    pos = int(np.log2(n))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(n, dtype=np.float64)
    for idx in range(n):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    n = N
    keich = _keich(n)
    I = np.arange(n, dtype=np.float64)
    i = np.repeat(I, n)
    j = np.tile(I, n)

    # ---- Phase 1: optimize 1-D union volume for x and y (product structure) ----
    def f1d(v):
        return _vol1d(v, n)

    def g1d(v):
        return -f1d(v)

    x_best = keich.copy()
    y_best = keich.copy()
    best_x = f1d(x_best)
    best_y = f1d(y_best)

    methods = [("L-BFGS-B", {}), ("Powell", {"maxiter": 400, "ftol": 1e-12})]
    for _ in range(3):
        if time.time() > DEADLINE - MARGIN:
            break
        for name, opts in methods:
            if time.time() > DEADLINE - MARGIN:
                break
            for v0 in (x_best + 0.02 * np.random.default_rng(0).normal(size=n), x_best):
                try:
                    res = minimize(g1d, v0, method=name, options=opts,
                                   tol=1e-12, maxiter=opts.get("maxiter", 200))
                    if res.success or (res.fun < -best_x + 1e-15):
                        fx = f1d(res.x)
                        if fx < best_x - 1e-15:
                            best_x = fx
                            x_best = res.x
                except Exception:
                    pass
            for v0 in (y_best + 0.02 * np.random.default_rng(1).normal(size=n), y_best):
                try:
                    res = minimize(g1d, v0, method=name, options=opts,
                                   tol=1e-12, maxiter=opts.get("maxiter", 200))
                    if res.success or (res.fun < -best_y + 1e-15):
                        fy = f1d(res.x)
                        if fy < best_y - 1e-15:
                            best_y = fy
                            y_best = res.x
                except Exception:
                    pass

    # build product candidate
    X = x_best
    Y = y_best
    P = np.stack([np.repeat(X, n), np.tile(Y, n)], axis=1)
    best_V = _vol2d(P, n)
    best_P = P.copy()

    # ---- Phase 2: full 2-D joint polish on all 64 positions (break product) ----
    def f2d(p):
        Pm = p.reshape(64, 2)
        return _vol2d(Pm, n)

    def g2d(p):
        return -f2d(p)

    p0 = best_P.flatten()
    for _ in range(2):
        if time.time() > DEADLINE - MARGIN:
            break
        for name, opts in methods:
            if time.time() > DEADLINE - MARGIN:
                break
            for pstart in (p0 + 0.02 * np.random.default_rng(2).normal(size=128), p0):
                try:
                    res = minimize(g2d, pstart, method=name, options=opts,
                                   tol=1e-12, maxiter=opts.get("maxiter", 300))
                    if res.success or (res.fun < -best_V + 1e-15):
                        v = f2d(res.x)
                        if v < best_V - 1e-15:
                            best_V = v
                            best_P = res.x.reshape(64, 2)
                except Exception:
                    pass

    # ---- Phase 3: high-precision local polish with tight tolerance ----
    if time.time() <= DEADLINE - MARGIN:
        try:
            res = minimize(g2d, best_P.flatten(), method="Powell",
                           options={"maxiter": 600, "ftol": 1e-14, "xtol": 1e-14},
                           tol=1e-14)
            if res.success:
                v = f2d(res.x)
                if v < best_V - 1e-15:
                    best_V = v
                    best_P = res.x.reshape(64, 2)
        except Exception:
            pass

    # ---- Validate with the exact evaluator MC; fall back to seed if invalid ----
    seed_P = np.stack([np.repeat(keich, n), np.tile(keich, n)], axis=1)
    V_seed = _vol_full(seed_P, n)
    V_cand = _vol_full(best_P, n)
    if (not np.isfinite(V_cand)) or V_cand <= 0:
        return seed_P
    if np.isfinite(V_seed) and V_seed > 0 and V_cand > V_seed:
        return seed_P
    return best_P

# EVOLVE-BLOCK-END
