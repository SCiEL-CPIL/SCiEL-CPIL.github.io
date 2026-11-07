"""Scale-invariant shape optimization for the second-autocorrelation constant.

The objective value = l2 / (norm_1 * norm_inf) is homogeneous of degree 0 in h
(all three quantities scale as h^2).  Under the normalization sum(h)=1 the
quantity norm_1 = (sum h)^2/(M+1) is a constant, so the objective reduces to
maximizing  l2 / norm_inf.  The "block" (rectangular) pulse produces a tent
autocorrelation with a flat plateau, which maximizes the width of the near-peak
region relative to the peak height -- exactly what l2/norm_inf rewards.  We do a
coarse scan over block widths, then refine the shape with Nelder-Mead, then
polish with L-BFGS-B using a smooth soft-max surrogate for the non-smooth
norm_inf.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_nelder_mead

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
_START = time.time()
_DEADLINE = _START + 140.0   # evaluator kills at 150 s; keep a 10 s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    denom = norm_1 * norm_inf
    if denom <= 0:
        return 0.0
    return l2 / denom


def _value_soft(h, tau):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    ag = np.abs(g)
    amax = np.max(ag)
    if amax <= 0:
        return 0.0
    norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
    denom = norm_1 * norm_inf_soft
    if denom <= 0:
        return 0.0
    return l2 / denom


def _value_grad(h, tau=0.0):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    y = np.zeros(M + 2)
    y[1:-1] = g
    y2 = y * y
    l2 = (W / 3.0) * np.sum(y2[:-1] + y[:-1] * y[1:] + y2[1:])
    norm_1 = np.sum(np.abs(g)) / (M + 1)

    if tau > 0:
        ag = np.abs(g)
        amax = np.max(ag)
        if amax <= 0:
            return 0.0, np.zeros_like(h)
        norm_inf_soft = tau * (np.log(np.sum(np.exp((ag - amax) / tau))) + amax / tau)
        wsoft = np.exp((ag - amax) / tau)
        wsoft /= np.sum(wsoft)
        dnh_dg = wsoft * np.sign(g)
        denom = norm_1 * norm_inf_soft
        val = l2 / denom
        norm_inf = norm_inf_soft
    else:
        norm_inf = np.max(np.abs(g))
        if norm_inf <= 0:
            return 0.0, np.zeros_like(h)
        imax = int(np.argmax(np.abs(g)))
        dnh_dg = np.zeros(M)
        dnh_dg[imax] = np.sign(g[imax])
        denom = norm_1 * norm_inf
        val = l2 / denom

    a = W / 3.0
    dly = np.zeros(M + 2)
    dly[:-1] += a * (2 * y[:-1] + y[1:])
    dly[1:]  += a * (y[:-1] + 2 * y[1:])
    dlg = dly[1:-1]

    conv1 = np.convolve(dlg, h, mode='full')
    conv2 = np.convolve(dlg[::-1], h, mode='full')
    dgh = conv1[:N] + conv2[N - 1:]

    dn1_dg = np.sign(g) / (M + 1)
    dnh1 = np.convolve(dn1_dg, h, mode='full')[:N] + np.convolve(dn1_dg[::-1], h, mode='full')[N - 1:]
    dnhinf = np.convolve(dnh_dg, h, mode='full')[:N] + np.convolve(dnh_dg[::-1], h, mode='full')[N - 1:]

    ddenom = dnh1 * norm_inf + norm_1 * dnhinf
    grad_val = dgh / denom - l2 * ddenom / (denom * denom)
    return val, grad_val


def _project(h):
    h = np.clip(h, 0.0, None)
    s = np.sum(h)
    if s <= 0:
        return np.ones(N) / N
    return h / s


def _nm_polish(h0, max_iter=4000):
    h = _project(h0.copy())
    v0 = _value(h)

    def neg_obj(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return 1e6
        return -_value(x / s)

    try:
        res = minimize_nelder_mead(neg_obj, h, maxiter=max_iter,
                                   xatol=1e-8, fatol=1e-9, adaptive=True,
                                   initial_simplex=0.01)
        h2 = _project(res.x)
        v2 = _value(h2)
        if v2 > v0:
            return h2, v2
    except Exception:
        pass
    return h, v0


def _lbfgs_polish(h0, max_iter=300):
    h = _project(h0.copy())
    tau = 1e-3

    def neg_obj(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return 1e6
        return -_value_soft(x / s, tau)

    def neg_grad(x):
        x = np.clip(x, 0, None)
        s = np.sum(x)
        if s <= 0:
            return np.zeros(N)
        _, gv = _value_soft(x / s, tau)
        return -gv

    try:
        res = minimize(neg_obj, h, method='L-BFGS-B', jac=neg_grad,
                       options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 20})
        h2 = _project(res.x)
        v2 = _value(h2)
        if v2 > _value(h):
            return h2, v2
    except Exception:
        pass
    return h, _value(h)


def _block(center, k):
    h = np.zeros(N)
    c = int(round(center))
    lo = max(0, c - k // 2)
    hi = min(N, lo + k)
    if hi > lo:
        h[lo:hi] = 1.0
    return h


def _optimize():
    best_h = None
    best_v = -1.0

    def consider(h0, nm_iter, lbfgs_iter):
        nonlocal best_h, best_v
        if _time_left() < 6:
            return
        h0 = np.clip(np.asarray(h0, float), 0, None)
        if not np.all(np.isfinite(h0)):
            return
        if np.sum(h0) <= 0:
            return
        h0 = h0 / np.sum(h0)
        h1, v1 = _nm_polish(h0, max_iter=nm_iter)
        if v1 > best_v:
            best_v = v1
            best_h = h1.copy()
        if _time_left() < 8:
            return
        h2, v2 = _lbfgs_polish(h1, max_iter=lbfgs_iter)
        if v2 > best_v:
            best_v = v2
            best_h = h2.copy()

    # ---- Stage 1: coarse scan over block widths (the tent-plateau family) ----
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 50]:
        if _time_left() < 20:
            break
        consider(_block(25, k), nm_iter=1500, lbfgs_iter=150)

    # ---- Stage 2: diverse non-block inits ----
    rng = np.random.default_rng(0)
    inits = [
        np.ones(N),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        rng.uniform(0, 1, N),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 3.0 ** 2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 5.0 ** 2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 8.0 ** 2)),
        np.exp(-((np.arange(N) - N / 2) ** 2) / (2 * 12.0 ** 2)),
        np.sin(np.pi * np.arange(N) / N) + 0.1,
        np.concatenate([np.ones(N // 2), np.zeros(N - N // 2)]),
        np.concatenate([np.zeros(N // 3), np.ones(N - 2 * (N // 3)), np.zeros(N // 3)]),
        np.concatenate([np.zeros(N // 4), np.linspace(0, 1, N - N // 4)]),
        np.abs(np.linspace(-1, 1, N)),
        # bimodal
        np.exp(-((np.arange(N) - 12) ** 2) / (2 * 3.0 ** 2)) + np.exp(-((np.arange(N) - 38) ** 2) / (2 * 3.0 ** 2)),
        # plateau
        np.concatenate([np.zeros(10), np.ones(30), np.zeros(10)]),
        # sharp center
        (lambda x: (x.__setitem__(slice(23, 27), 1.0), x)[1])(np.zeros(N)),
        (lambda x: (x.__setitem__(slice(24, 26), 1.0), x)[1])(np.zeros(N)),
    ]
    for h0 in inits:
        if _time_left() < 15:
            break
        consider(h0, nm_iter=1200, lbfgs_iter=150)

    # ---- Stage 3: final heavy polish from the global best ----
    if best_h is not None and _time_left() > 12:
        h3, v3 = _nm_polish(best_h, max_iter=6000)
        if v3 > best_v:
            best_v = v3
            best_h = h3.copy()
        if _time_left() > 8:
            h4, v4 = _lbfgs_polish(best_h, max_iter=400)
            if v4 > best_v:
                best_v = v4
                best_h = h4.copy()

    if best_h is None:
        best_h = np.ones(N)
        best_v = _value(best_h)
    return best_h, best_v


def construct():
    try:
        h, v = _optimize()
        h = np.asarray(h, float)
        h = np.clip(h, 0, None)
        if not np.all(np.isfinite(h)) or np.sum(h) <= 0:
            h = np.ones(N)
        return list(h)
    except Exception:
        return [1.0] * N
