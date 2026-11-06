"""Broadband antireflection coating design via characteristic-matrix + L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0 - 8.0  # wall-clock safety margin


def _reflectance(labels, thicknesses, n_amb, n_sub, n_mat, wl):
    """Mean normal-incidence reflectance (%) over wl (vectorized char-matrix)."""
    nL = n_mat['L']; nH = n_mat['H']
    n = np.where(np.asarray(labels) == 'H', nH, nL)
    d = np.asarray(thicknesses, dtype=float)
    w = np.asarray(wl, dtype=float)
    # phase beta_j = 2*pi*n_j*d_j / wl   (shape len(wl), N)
    beta = 2.0 * np.pi * (n[None, :] * d[None, :]) / w[:, None]
    c = np.cos(beta); s = np.sin(beta)
    # characteristic matrices M_j = [[c, i s / n_j], [i n_j s, c]]
    A = np.ones((w.size, 2, 2)); B = np.ones((w.size, 2, 2))
    A[:, 0, 0] = c; A[:, 0, 1] = 1j * s / n[None, :]
    A[:, 1, 0] = 1j * n[None, :] * s; A[:, 1, 1] = c
    M = A @ B
    r0 = n_amb; r1 = n_sub
    M11 = M[:, 0, 0]; M12 = M[:, 0, 1]; M21 = M[:, 1, 0]; M22 = M[:, 1, 1]
    Y = (M11 * r1 + M12) / (M21 * r1 + M22)
    r = (r0 - Y) / (r0 + Y)
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def design(problem):
    band = problem["band_nm"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_mat = problem["materials"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    wl = np.linspace(band[0], band[1], 401)
    labels = ['H' if (i % 2 == 0) else 'L' for i in range(max_layers)]
    bounds = [(0.0, float(max_layer))] * max_layers

    def obj(x):
        return _reflectance(labels, x, n_amb, n_sub, n_mat, wl)

    start = time.time()
    x0 = np.full(max_layers, 100.0)

    def cb(xk):
        if time.time() > _DEADLINE:
            raise RuntimeError("deadline")

    best_x = x0.copy(); best_f = obj(x0)
    try:
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=3000, maxfun=20000,
                                    ftol=1e-12, gtol=1e-10, disp=False),
                       callback=cb)
        if res.fun < best_f:
            best_f = res.fun; best_x = res.x
    except Exception:
        pass

    # enforce total-thickness constraint
    if float(np.sum(best_x)) > max_total:
        best_x = best_x * (max_total / float(np.sum(best_x)))
    best_x = np.clip(best_x, 0.0, max_layer)

    return labels, [float(t) for t in best_x]
