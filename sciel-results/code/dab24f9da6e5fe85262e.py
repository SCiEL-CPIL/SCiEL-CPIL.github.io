# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

_N = 64
_K = int(np.log2(_N))
_A = (np.arange(_N, dtype=np.float64) + 1.0) / _N
_W = 1.0 / _N
_Y = (np.arange(20000, dtype=np.float64) + 0.5) / 20000.0
_AI = _A[:, None]
_AJ = _A[None, :]
_D = _AJ - _AI
_DW = _D * _W


def _keich_init():
    n = _N
    k = _K
    j = np.arange(1, k + 1, dtype=np.float64)
    coef = (1.0 - j) / k * (2.0 ** (-j))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        eps = np.array([int(b) for b in format(i, '0%db' % k)], dtype=np.float64)
        out[i] = float(np.dot(coef, eps))
    return out


def _area_and_grad(x):
    L = x[:, None] + _AI * _Y[None, :]
    Ls = np.sort(L, axis=0)
    diffs = Ls[:, 1:] - Ls[:, :-1]
    m = np.minimum(diffs, _W)
    area = float(np.mean(_W + m.sum(axis=1)))
    S = np.argsort(L, axis=0)
    I = np.arange(_N)
    G = np.zeros_like(L)
    for j in range(_N - 1):
        a = S[:, j]
        b = S[:, j + 1]
        d = diffs[:, j]
        mask = d < _W
        c = _D[np.ix_(a, b)]
        G[a, mask] += c[mask]
        G[b, mask] -= c[mask]
    grad = G.mean(axis=1)
    return area, grad


def _area_only(x):
    L = x[:, None] + _AI * _Y[None, :]
    Ls = np.sort(L, axis=0)
    diffs = Ls[:, 1:] - Ls[:, :-1]
    return float(np.mean(_W + np.minimum(diffs, _W).sum(axis=1)))


def _subgrad(x, deadline):
    x = np.array(x, dtype=np.float64).copy()
    f0, g = _area_and_grad(x)
    t0 = 0.05
    best_f = f0
    best_x = x.copy()
    step = 0
    while time.time() < deadline:
        ng = float(np.linalg.norm(g))
        step += 1
        if ng < 1e-12:
            break
        g_hat = g / ng
        c = 1e-4
        t = t0
        for _ in range(60):
            x_try = x - t * g_hat
            f_try, _ = _area_and_grad(x_try)
            if f_try <= f0 - c * t * ng:
                break
            t *= 0.5
            if t < 1e-14:
                break
        x = x - t * g_hat
        f0, g = _area_and_grad(x)
        if f0 < best_f:
            best_f = f0
            best_x = x.copy()
        if t < 1e-13:
            break
        t0 = t
    return best_x, best_f


def _coord_descent(x, deadline, tol=1e-12, max_inner=60):
    x = np.array(x, dtype=np.float64).copy()
    f0 = _area_only(x)
    best_f = f0
    best_x = x.copy()
    i = 0
    while time.time() < deadline:
        idx = i % _N
        i += 1
        old = x[idx]
        lo = old - 0.5
        hi = old + 0.5
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a = lo
        b = hi
        c_ = b - gr * (b - a)
        d_ = a + gr * (b - a)
        x[idx] = c_
        fc = _area_only(x)
        x[idx] = d_
        fd = _area_only(x)
        for _ in range(max_inner):
            if fc < fd:
                b = d_
                d_ = c_
                fd = fc
                c_ = b - gr * (b - a)
                x[idx] = c_
                fc = _area_only(x)
            else:
                a = c_
                c_ = d_
                fc = fd
                d_ = a + gr * (b - a)
                x[idx] = d_
                fd = _area_only(x)
            if b - a < tol:
                break
            if time.time() > deadline:
                break
        x[idx] = (a + b) / 2.0
        fnew = _area_only(x)
        if fnew < best_f:
            best_f = fnew
            best_x = x.copy()
        f0 = fnew
    return best_x, best_f


def _smooth_area_and_grad(x, beta=50.0):
    """Smooth surrogate: replace min(d, W) with soft-min via log-sum-exp."""
    L = x[:, None] + _AI * _Y[None, :]
    Ls = np.sort(L, axis=0)
    diffs = Ls[:, 1:] - Ls[:, :-1]
    # soft_min(d, W) = W - soft_max(0, W - d)
    # soft_max(0, u) = (1/beta) * logaddexp(0, beta*u)
    u = _W - diffs
    u_clipped = np.clip(u, -50.0, 50.0)
    soft_max_0_u = np.logaddexp(0.0, beta * u_clipped) / beta
    soft_min = _W - soft_max_0_u
    area = float(np.mean(_W + soft_min.sum(axis=1)))
    # Gradient: d(soft_min)/dx_i via chain rule
    S = np.argsort(L, axis=0)
    G = np.zeros_like(L)
    for j in range(_N - 1):
        a = S[:, j]
        b = S[:, j + 1]
        d = diffs[:, j]
        sig = 1.0 / (1.0 + np.exp(-beta * (_W - d)))
        c = _D[np.ix_(a, b)]
        G[a, :] -= sig * c
        G[b, :] += sig * c
    grad = G.mean(axis=1)
    return area, grad


def _lbfgs_smooth(x, deadline, beta=50.0):
    x = np.array(x, dtype=np.float64).copy()
    try:
        res = _scipy_minimize(
            lambda v: _smooth_area_and_grad(v, beta)[0],
            x.copy(),
            jac=lambda v: _smooth_area_and_grad(v, beta)[1],
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-8, 'ftol': 1e-14},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            return np.array(res.x, dtype=np.float64)
    except Exception:
        pass
    return x


def _lbfgs_polish(x, deadline):
    x = np.array(x, dtype=np.float64).copy()
    try:
        res = _scipy_minimize(
            lambda v: _area_and_grad(v)[0],
            x.copy(),
            jac=lambda v: _area_and_grad(v)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-9, 'ftol': 1e-14},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            return np.array(res.x, dtype=np.float64)
    except Exception:
        pass
    return x


def kakeya_needle_2d_positions64() -> np.ndarray:
    start = time.time()
    hard_deadline = start + 600.0
    # Reserve margin: do all work before this.
    deadline = hard_deadline - 30.0

    x = _keich_init()
    f = _area_only(x)
    best_x = x.copy()
    best_f = f

    # Phase 1: subgradient descent
    try:
        x1, f1 = _subgrad(x, deadline)
        if f1 < best_f:
            best_f = f1
            best_x = x1.copy()
        x = x1
    except Exception:
        pass

    # Phase 2: coordinate descent (refine each coordinate to high precision)
    try:
        x2, f2 = _coord_descent(x, deadline)
        if f2 < best_f:
            best_f = f2
            best_x = x2.copy()
        x = x2
    except Exception:
        pass

    # Phase 3: Smooth L-BFGS (handles nonsmoothness via soft-min surrogate)
    if time.time() < deadline:
        try:
            x3 = _lbfgs_smooth(x, deadline, beta=50.0)
            f3 = _area_only(x3)
            if f3 < best_f and np.all(np.isfinite(x3)):
                best_f = f3
                best_x = x3.copy()
                x = x3
        except Exception:
            pass

    # Phase 4: Exact L-BFGS polish
    if time.time() < deadline:
        try:
            x4 = _lbfgs_polish(x, deadline)
            f4 = _area_only(x4)
            if f4 < best_f and np.all(np.isfinite(x4)):
                best_f = f4
                best_x = x4.copy()
                x = x4
        except Exception:
            pass

    # Phase 5: Final high-precision coordinate descent
    if time.time() < deadline:
        try:
            x5, f5 = _coord_descent(x, deadline, tol=1e-14, max_inner=80)
            if f5 < best_f:
                best_f = f5
                best_x = x5.copy()
        except Exception:
            pass

    # Safety: ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_init()

    return best_x


# EVOLVE-BLOCK-END
