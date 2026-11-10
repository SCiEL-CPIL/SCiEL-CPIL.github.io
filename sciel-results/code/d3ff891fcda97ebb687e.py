# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 8
w = 1.0 / N
i_vals = np.arange(N, dtype=np.float64)
j_vals = np.arange(N, dtype=np.float64)

# Precompute coefficients for the 1-D union length:
# len(union of [x_i + c*i/N, x_i + c*i/N + 1/N]) = (1/N) * sum_i (1 - c*i/N)
coef_i = i_vals / N
coef_j = j_vals / N


def _union_len_1d(x, c, coef):
    """Length of union of N intervals [x_i + c*coef_i, x_i + c*coef_i + w] at height c."""
    lo = x[:, None] + c[None, :] * coef[:, None]
    hi = lo + w
    order = np.argsort(lo, axis=0)
    s_lo = np.take_along_axis(lo, order, axis=0)
    s_hi = np.take_along_axis(hi, order, axis=0)
    cummax = np.maximum.accumulate(s_hi, axis=0)
    contrib = np.maximum(0.0, s_hi - np.maximum(s_lo, np.concatenate([np.zeros((1, s_lo.shape[1])), cummax[:-1]], axis=0)))
    return contrib.sum(axis=0)


def _compute_volume_exact(x, y):
    """Exact union volume via product structure + trapezoidal integration."""
    n = 200
    c = np.linspace(0.0, 1.0, n)
    len_a = _union_len_1d(x, c, coef_i)
    len_b = _union_len_1d(y, c, coef_j)
    area = len_a * len_b
    vol = np.trapz(area, c)
    x0 = x.min()
    x1 = (x + coef_i).max() + w
    y0 = y.min()
    y1 = (y + coef_j).max() + w
    if (x1 - x0) * (y1 - y0) <= 1e-9:
        return np.inf
    return vol


def _objective_1d(x):
    """1-D objective: integral of (union length)^2 / (bounding length)."""
    n = 150
    c = np.linspace(0.0, 1.0, n)
    L = _union_len_1d(x, c, coef_i)
    box = (x + coef_i).max() + w - x.min()
    if box <= 1e-9:
        return 1e10
    return np.trapz(L * L, c) / box


def kakeya_needle_3d8() -> np.ndarray:
    import time
    t0 = time.time()
    deadline = t0 + 600.0
    safety = 30.0

    # Keich binary construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x0 = keich.copy()
    y0 = keich.copy()

    best_x = x0.copy()
    best_y = y0.copy()
    best_vol = _compute_volume_exact(best_x, best_y)

    # Optimize x
    try:
        res_x = minimize(_objective_1d, x0.copy(), method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if res_x.success or res_x.fun < _objective_1d(x0):
            x_cand = res_x.x
            vol_cand = _compute_volume_exact(x_cand, y0)
            if vol_cand < best_vol:
                best_x = x_cand
                best_vol = vol_cand
    except Exception:
        pass

    if time.time() > deadline - safety:
        return np.stack([best_x[i_vals // 1], best_y], axis=1)  # fallback shape fix below

    # Optimize y
    try:
        res_y = minimize(_objective_1d, y0.copy(), method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if res_y.success or res_y.fun < _objective_1d(y0):
            y_cand = res_y.x
            vol_cand = _compute_volume_exact(best_x, y_cand)
            if vol_cand < best_vol:
                best_y = y_cand
                best_vol = vol_cand
    except Exception:
        pass

    # Second pass: re-optimize x given new y, then y given new x
    if time.time() < deadline - safety:
        try:
            res_x2 = minimize(_objective_1d, best_x.copy(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res_x2.fun < _objective_1d(best_x):
                x_cand = res_x2.x
                vol_cand = _compute_volume_exact(x_cand, best_y)
                if vol_cand < best_vol:
                    best_x = x_cand
                    best_vol = vol_cand
        except Exception:
            pass

    if time.time() < deadline - safety:
        try:
            res_y2 = minimize(_objective_1d, best_y.copy(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if res_y2.fun < _objective_1d(best_y):
                y_cand = res_y2.x
                vol_cand = _compute_volume_exact(best_x, y_cand)
                if vol_cand < best_vol:
                    best_y = y_cand
                    best_vol = vol_cand
        except Exception:
            pass

    # Build 64x2 positions: row k, i=k//8, j=k%8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions

# EVOLVE-BLOCK-END
