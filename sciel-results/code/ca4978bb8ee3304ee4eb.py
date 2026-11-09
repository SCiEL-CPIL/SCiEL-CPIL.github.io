# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_N = 8
_N2 = _N * _N
_I = np.arange(_N2) // _N
_J = np.arange(_N2) % _N
_INV_N = 1.0 / _N

def _union_1d(intervals):
    """intervals: (M,2) array of [lo,hi]. Return total length of union."""
    if intervals.shape[0] == 0:
        return 0.0
    order = np.argsort(intervals[:, 0])
    lo = intervals[order, 0]
    hi = intervals[order, 1]
    total = 0.0
    cur_lo = lo[0]
    cur_hi = hi[0]
    for k in range(1, len(lo)):
        if lo[k] <= cur_hi:
            if hi[k] > cur_hi:
                cur_hi = hi[k]
        else:
            total += cur_hi - cur_lo
            cur_lo = lo[k]
            cur_hi = hi[k]
    total += cur_hi - cur_lo
    return total

def _obj_1d(x, n_quad=50):
    """
    Objective for the 1D problem: minimize the average union length of 8 intervals.
    x: shape (8,)
    """
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    # x_i + c * i / 8
    # i is 0..7
    i_idx = np.arange(_N)
    # shifts shape (n_quad, 8)
    shifts = np.outer(c, i_idx * _INV_N)
    lo = x[None, :] + shifts
    hi = lo + _INV_N
    # intervals for each c: shape (n_quad, 8, 2)
    intervals = np.stack([lo, hi], axis=2)
    
    total_len = 0.0
    for k in range(n_quad):
        total_len += w[k] * 0.5 * _union_1d(intervals[k])
    return total_len

def _exact_volume(params, n_quad=24):
    """Exact union volume via Gauss-Legendre quadrature in c in [0,1]."""
    x = params[0:_N2]
    y = params[_N2:2 * _N2]
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    vol = 0.0
    for ck, wk in zip(c, w):
        xoff = ck * _I * _INV_N
        yoff = ck * _J * _INV_N
        xl = x + xoff
        xr = xl + _INV_N
        yl = y + yoff
        yr = yl + _INV_N
        lx = _union_1d(np.stack([xl, xr], axis=1))
        ly = _union_1d(np.stack([yl, yr], axis=1))
        vol += wk * 0.5 * lx * ly
    return vol

def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    deadline = start + 580.0

    def time_left():
        return deadline - time.time()

    # Step 1: Solve the 1D problem to find optimal x positions
    # The 1D problem is 8-dimensional.
    # Initial guess: uniform spread or Keich-like
    # Keich init for 1D:
    N = _N
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_1d_vol = np.inf
    best_1d_x = keich.copy()
    
    rng = np.random.default_rng(42)
    # Try multiple starts for the 1D problem
    starts = [keich.copy(), np.zeros(N), rng.normal(0, 0.5, N)]
    for s in starts:
        if time_left() < 10.0:
            break
        try:
            res = minimize(
                _obj_1d,
                s,
                method="Nelder-Mead",
                options={
                    "maxiter": 5000,
                    "xatol": 1e-10,
                    "fatol": 1e-10,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_1d_vol:
                best_1d_vol = res.fun
                best_1d_x = res.x.copy()
        except Exception:
            pass

    # Step 2: Initialize the 128D problem with the best 1D solution for both X and Y
    # x_i depends on i, y_j depends on j.
    # In the 128D param vector, x_k is the base x for tube (i,j) where i=k//8.
    # So x_k should be best_1d_x[i].
    # y_k should be best_1d_x[j].
    p0 = np.zeros(2 * _N2)
    for k in range(_N2):
        i = k // _N
        j = k % _N
        p0[k] = best_1d_x[i]
        p0[k + _N2] = best_1d_x[j]
    
    best_params = p0.copy()
    best_vol = _exact_volume(best_params)

    # Step 3: Refine the 128D parameters
    # Use Powell or Nelder-Mead on the full space, starting from the product solution.
    # The full space is 128D, which is high for Nelder-Mead. Powell is better.
    # But Powell is slow.
    # We can try a few iterations of Powell.
    
    if time_left() > 30.0:
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Powell",
                options={
                    "maxiter": 100,
                    "maxfev": 2000,
                    "ftol": 1e-12,
                    "xtol": 1e-12,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    x = best_params[0:_N2]
    y = best_params[_N2:2 * _N2]
    positions = np.stack([x, y], axis=1)
    return positions

# EVOLVE-BLOCK-END
