"""Optimize step heights to minimize max overlap of h with shifted 1-h."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _autocorr(h):
    """Circular autocorrelation r[k] = sum_i h[i] h[(i+k) mod N]."""
    return np.real(np.fft.ifft(np.abs(np.fft.fft(h)) ** 2))


def _overlap_value(h):
    """Max over non-zero circular shifts of sum_i h[i]*(1-h[i+k])."""
    r = _autocorr(h)
    return float(np.max(N / 2.0 - r[1:]))


def _fix_sum(h):
    """Clip to [0,1] and make sum exactly N/2 by adjusting slack elements."""
    h = np.clip(h, 0.0, 1.0)
    res = N / 2.0 - h.sum()
    if res > 0:
        idx = np.argsort(1.0 - h)
    elif res < 0:
        idx = np.argsort(h)
    else:
        return h
    for i in idx:
        if res > 0:
            room = 1.0 - h[i]
            step = min(room, res)
            h[i] += step
            res -= step
        else:
            room = h[i]
            step = min(room, -res)
            h[i] -= step
            res += step
        if abs(res) < 1e-15:
            break
    return h


def construct():
    t0 = time.time()
    DEADLINE = 115.0

    best_h = np.full(N, 0.5)
    best_val = _overlap_value(best_h)  # 25.0

    def softmin_obj(h, Ks):
        r = _autocorr(h)
        neg = -r[1:]
        m = np.max(neg)
        return (m + np.log(np.sum(np.exp(neg - m)))) / Ks

    def softmin_grad(h, Ks):
        r = _autocorr(h)
        neg = -r[1:]
        m = np.max(neg)
        e = np.exp(neg - m)
        Z = np.sum(e)
        w = e / Z  # weights over k = 1..N-1
        g = np.zeros(N)
        for k in range(1, N):
            g += w[k - 1] * (np.roll(h, k) + np.roll(h, -k))
        return -g / Ks

    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq',
             'fun': lambda h: np.sum(h) - N / 2.0,
             'jac': lambda h: np.ones(N)}]

    rng = np.random.default_rng(0)
    inits = [np.full(N, 0.5)]
    for _ in range(5):
        h0 = rng.uniform(0.0, 1.0, N)
        h0 = h0 - 0.5
        h0 = h0 - h0.mean()
        h0 = 0.5 + 0.5 * h0
        h0 = _fix_sum(h0)
        inits.append(h0)

    Ks_schedule = [10.0, 40.0, 120.0, 300.0]

    for h0 in inits:
        h = h0.copy()
        for Ks in Ks_schedule:
            if time.time() > DEADLINE - 25.0:
                break
            try:
                res = minimize(softmin_obj, h,
                               jac=lambda x, K=Ks: softmin_grad(x, K),
                               method='SLSQP', bounds=bounds,
                               constraints=cons,
                               options={'maxiter': 150, 'ftol': 1e-12})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    h = res.x
            except Exception:
                pass
        # projected-gradient polish to tighten the soft-min toward true min
        for _ in range(150):
            if time.time() > DEADLINE - 25.0:
                break
            grad = softmin_grad(h, 300.0)
            h = h - 1e-3 * grad
            h = _fix_sum(h)
        val = _overlap_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    best_h = _fix_sum(best_h)
    # safety: never worse than the constant seed
    if _overlap_value(best_h) > 25.0:
        best_h = np.full(N, 0.5)
    return best_h
