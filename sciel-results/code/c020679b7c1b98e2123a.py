# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_area(x, M=20000):
    n = 64
    a = (np.arange(n) + 1.0) / n
    w = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    diffs = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(diffs, w), axis=1)
    return float(np.mean(union_len))

def _compute_grad(x, M=20000):
    n = 64
    a = (np.arange(n) + 1.0) / n
    w = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    diffs = Ls[:, 1:] - Ls[:, :-1]
    contrib = np.minimum(diffs, w)
    union_len = w + contrib.sum(axis=1)
    active = contrib > 0
    g = np.zeros((M, n), dtype=np.float64)
    g[:, 0] = 1.0
    g[:, 1:] = active - np.roll(active, 1, axis=1)
    grad = (g @ a[None, :]).sum(axis=0) / M
    return grad

def _slsqp_optimize(x0, max_iter=200):
    from scipy.optimize import minimize
    x0 = np.asarray(x0, dtype=np.float64)
    res = minimize(
        fun=lambda x: _compute_area(x),
        x0=x0,
        jac=lambda x: _compute_grad(x),
        method='SLSQP',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
    )
    return res.x, res.fun

def _multi_start_optimize(x_keich, time_limit, max_starts=8):
    best_x = x_keich.copy()
    best_area = _compute_area(best_x)
    t_start = time.time()
    rng = np.random.default_rng(42)
    for s in range(max_starts):
        if time.time() - t_start > time_limit:
            break
        if s == 0:
            x0 = x_keich.copy()
        else:
            x0 = x_keich + rng.normal(0, 0.05, size=64)
        x_opt, area_opt = _slsqp_optimize(x0)
        if area_opt < best_area:
            best_area = area_opt
            best_x = x_opt.copy()
    return best_x, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_keich = np.array(positions, dtype=np.float64)

    t_start = time.time()
    time_limit = 580.0

    best_x, best_area = _multi_start_optimize(x_keich, time_limit)

    # Final safety: ensure finite
    if not np.all(np.isfinite(best_x)):
        return x_keich
    return best_x

# EVOLVE-BLOCK-END
