import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_n = np.arange(N)


def _make_h(coeffs, M):
    # Band-limited real signal with unit sum:
    # h[n] = (1 + sum_{j>=1} a_j cos(2 pi j n/N) + b_j sin(2 pi j n/N)) / N
    h = np.full(N, 1.0 / N)
    for j in range(1, M + 1):
        a = coeffs[2 * (j - 1)]
        b = coeffs[2 * (j - 1) + 1]
        h += (a * np.cos(2 * np.pi * j * _n / N) + b * np.sin(2 * np.pi * j * _n / N)) / N
    return h


def _obj(coeffs, M):
    h = _make_h(coeffs, M)
    s = h.sum()
    if abs(s) < 1e-12:
        return 1e12
    c = np.convolve(h, h)
    return 2.0 * N * c.max() / (s * s)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under the 120s / 150s limits

    best_h = np.ones(N)
    best_val = 2.0
    best_coeffs = None
    best_M = None

    def upd(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        s = h.sum()
        if abs(s) < 1e-12:
            return
        v = abs(2.0 * N * np.convolve(h, h).max() / (s * s))
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Structured seeds (zero-mean, unit-sum after scaling handled by upd).
    def norm(x):
        x = x - x.mean()
        m = np.abs(x).max()
        return x / (m + 1e-12)

    seeds = [norm(np.cos(2 * np.pi * _n * _n / (2 * N))),
             norm(np.where(_n % 2 == 0, 1.0, -1.0))]
    for f in [1, 2, 3, 5, 8, 13]:
        seeds.append(norm(np.cos(2 * np.pi * f * _n / N)))
    for x in seeds:
        upd(x)

    # Global search over a small number of Fourier coefficients.
    plan = [(3, 250, 18), (5, 150, 18), (8, 100, 16),
            (12, 60, 15), (16, 40, 14), (24, 25, 12)]
    for M, maxiter, popsize in plan:
        if time.time() > deadline:
            break
        dim = 2 * M
        bounds = [(-3.0, 3.0)] * dim
        init = 'sobol' if dim <= 20 else 'latinhypercube'
        try:
            res = differential_evolution(
                _obj, bounds, args=(M,),
                maxiter=maxiter, popsize=popsize,
                tol=0.05, mutation=(0.4, 1.0), recombination=0.8,
                seed=12345, polish=False, init=init)
            if res.x is not None:
                v = _obj(res.x, M)
                h = _make_h(res.x, M)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
                    best_coeffs = res.x.copy()
                    best_M = M
                else:
                    upd(h)
        except Exception:
            pass

    # Local polish of the best coefficient vector.
    if best_coeffs is not None and time.time() < deadline - 8:
        try:
            r2 = minimize(_obj, best_coeffs, args=(best_M,), method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8})
            if r2.x is not None:
                upd(_make_h(r2.x, best_M))
        except Exception:
            pass

    return best_h
