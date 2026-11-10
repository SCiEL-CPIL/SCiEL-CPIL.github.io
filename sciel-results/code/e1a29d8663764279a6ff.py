# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _union_length(L, w):
    """Union length of unit-width-ish intervals [L_j, L_j + w] given sorted L."""
    Ls = np.sort(L)
    d = np.diff(Ls)
    return w + np.sum(np.minimum(d, w))


def _area(x):
    """Exact quadrature area of the union of the 64 triangles."""
    n = 64
    a = (np.arange(n) + 1.0) / n
    w = 1.0 / n
    y = (np.arange(20000) + 0.5) / 20000.0
    L = x[None, :] + a[None, :] * y[:, None]
    return np.mean(_union_length(L, w))


def _soft_area(x, lam):
    """Smooth log-sum-exp surrogate of the union area (for gradient descent)."""
    n = 64
    a = (np.arange(n) + 1.0) / n
    w = 1.0 / n
    y = (np.arange(20000) + 0.5) / 20000.0
    L = x[None, :] + a[None, :] * y[:, None]
    d = L[:, 1:] - L[:, :-1]
    # soft minimum of d and w
    soft = -lam * np.logaddexp(-d / lam, -w / lam)
    return np.mean(w + np.sum(soft, axis=1))


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich = np.array([
        np.sum((1 - j) * np.array(list(str(bin(i))[2:]), dtype=float)
               if len(str(bin(i))[2:]) == k
               else (1 - j) * np.pad(np.array(list(str(bin(i))[2:]), dtype=float),
                                     (k - len(str(bin(i))[2:]), 0)) * (2.0 ** -j) / k)
        for i in range(n)
    ], dtype=np.float64)
    # simpler robust keich
    keich = np.zeros(n)
    for i in range(n):
        bits = list(str(bin(i))[2:])
        eps = np.pad(np.array(bits, dtype=float), (k - len(bits), 0))
        keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)

    best_x = keich.copy()
    best_area = _area(keich)

    t0 = time.time()
    deadline = 600.0 - 30.0  # safety margin

    # ---- Annealed smooth surrogate refinement from the Keich seed ----
    lam = 1.0
    x = best_x.copy()
    while lam > 1e-6 and (time.time() - t0) < deadline:
        res = minimize(
            _soft_area, x, args=(lam,),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 20000, 'gtol': 1e-8},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = res.x
            ca = _area(cand)
            if ca < best_area:
                best_area = ca
                best_x = cand
            x = cand
        lam *= 0.5

    # ---- Multi-start random restarts with short annealing ----
    rng = np.random.default_rng(20240517)
    for _ in range(200):
        if (time.time() - t0) > deadline:
            break
        x0 = best_x + rng.normal(0.0, 0.02, size=n)
        lam = 0.5
        x = x0
        while lam > 1e-5 and (time.time() - t0) < deadline:
            res = minimize(
                _soft_area, x, args=(lam,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 5000, 'gtol': 1e-7},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                x = res.x
            lam *= 0.5
        ca = _area(x)
        if ca < best_area:
            best_area = ca
            best_x = x

    # ---- Final local polish on the exact objective ----
    if (time.time() - t0) < deadline:
        res = minimize(
            _area, best_x, method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            ca = _area(res.x)
            if ca < best_area:
                best_area = ca
                best_x = res.x

    return np.asarray(best_x, dtype=np.float64)


# EVOLVE-BLOCK-END
