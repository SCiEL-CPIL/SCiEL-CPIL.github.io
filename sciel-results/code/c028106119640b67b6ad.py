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

def _exact_volume(params, n_quad=60):
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

def _exact_volume_fast(params, n_quad=60):
    """Fast version using vectorized union where possible, fallback to loop."""
    x = params[0:_N2]
    y = params[_N2:2 * _N2]
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    vol = 0.0
    for k in range(n_quad):
        ck = c[k]
        wk = w[k]
        xoff = ck * _I * _INV_N
        yoff = ck * _J * _INV_N
        xl = x + xoff
        xr = xl + _INV_N
        yl = y + yoff
        yr = yl + _INV_N
        # Sort intervals by lo
        x_order = np.argsort(xl)
        xlo = xl[x_order]
        xhi = xr[x_order]
        y_order = np.argsort(yl)
        ylo = yl[y_order]
        yhi = yr[y_order]
        
        # Union length computation (vectorized-ish)
        def _union_len(lo, hi):
            total = 0.0
            cur_hi = hi[0]
            for idx in range(1, len(lo)):
                if lo[idx] <= cur_hi:
                    if hi[idx] > cur_hi:
                        cur_hi = hi[idx]
                else:
                    total += cur_hi - lo[idx-1] if idx > 0 else 0
                    # Actually need to track cur_lo too
                    break
            # Let me just use the loop version for correctness
            return 0.0
        
        lx = _union_1d(np.stack([xl, xr], axis=1))
        ly = _union_1d(np.stack([yl, yr], axis=1))
        vol += wk * 0.5 * lx * ly
    return vol

def _obj_1d(x, n_quad=80):
    """
    Objective for the 1D problem: minimize the average union length of 8 intervals.
    x: shape (8,)
    """
    t, w = np.polynomial.legendre.leggauss(n_quad)
    c = 0.5 * (t + 1.0)
    i_idx = np.arange(_N)
    shifts = np.outer(c, i_idx * _INV_N)
    lo = x[None, :] + shifts
    hi = lo + _INV_N
    intervals = np.stack([lo, hi], axis=2)
    
    total_len = 0.0
    for k in range(n_quad):
        total_len += w[k] * 0.5 * _union_1d(intervals[k])
    return total_len

def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0

    def time_left():
        return deadline - time.time()

    # Step 1: Solve the 1D problem to find optimal x positions
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
    # Generate diverse starting points
    starts = [keich.copy(), np.zeros(N), rng.normal(0, 0.5, N)]
    for shift in [-0.3, -0.1, 0.1, 0.3]:
        starts.append(keich.copy() + shift)
    for scale in [0.5, 1.5, 2.0]:
        starts.append(keich.copy() * scale)
    
    for s in starts:
        if time_left() < 20.0:
            break
        try:
            res = minimize(
                _obj_1d,
                s,
                method="Nelder-Mead",
                options={
                    "maxiter": 10000,
                    "xatol": 1e-14,
                    "fatol": 1e-14,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_1d_vol:
                best_1d_vol = res.fun
                best_1d_x = res.x.copy()
        except Exception:
            pass

    # Step 2: Initialize the 128D problem with the best 1D solution
    p0 = np.zeros(2 * _N2)
    for k in range(_N2):
        i = k // _N
        j = k % _N
        p0[k] = best_1d_x[i]
        p0[k + _N2] = best_1d_x[j]
    
    best_params = p0.copy()
    best_vol = _exact_volume(best_params)
    
    # Step 3: Coordinate descent on x and y separately with multiple restarts
    if time_left() > 80.0:
        # Optimize x coordinates
        x0 = best_params[0:_N2].copy()
        y0 = best_params[_N2:2 * _N2].copy()
        
        def obj_x(x):
            return _exact_volume(np.concatenate([x, y0]))
        
        try:
            res_x = minimize(
                obj_x,
                x0,
                method="Powell",
                options={
                    "maxiter": 100,
                    "maxfev": 3000,
                    "ftol": 1e-15,
                    "xtol": 1e-15,
                    "disp": False,
                },
            )
            if res_x.fun < best_vol:
                best_vol = res_x.fun
                best_params[0:_N2] = res_x.x
        except Exception:
            pass

        # Optimize y coordinates
        x1 = best_params[0:_N2].copy()
        y1 = best_params[_N2:2 * _N2].copy()
        
        def obj_y(y):
            return _exact_volume(np.concatenate([x1, y]))
        
        if time_left() > 40.0:
            try:
                res_y = minimize(
                    obj_y,
                    y1,
                    method="Powell",
                    options={
                        "maxiter": 100,
                        "maxfev": 3000,
                        "ftol": 1e-15,
                        "xtol": 1e-15,
                        "disp": False,
                    },
                )
                if res_y.fun < best_vol:
                    best_vol = res_y.fun
                    best_params[_N2:2 * _N2] = res_y.x
            except Exception:
                pass

    # Step 4: Multiple joint polish passes with different methods
    if time_left() > 30.0:
        # Powell polish
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Powell",
                options={
                    "maxiter": 50,
                    "maxfev": 1000,
                    "ftol": 1e-16,
                    "xtol": 1e-16,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Step 5: Nelder-Mead final polish
    if time_left() > 15.0:
        try:
            res = minimize(
                _exact_volume,
                best_params,
                method="Nelder-Mead",
                options={
                    "maxiter": 5000,
                    "xatol": 1e-15,
                    "fatol": 1e-15,
                    "adaptive": True,
                    "disp": False,
                },
            )
            if res.fun < best_vol:
                best_vol = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Step 6: One more coordinate descent cycle if time allows
    if time_left() > 25.0:
        x0 = best_params[0:_N2].copy()
        y0 = best_params[_N2:2 * _N2].copy()
        
        def obj_x2(x):
            return _exact_volume(np.concatenate([x, y0]))
        
        try:
            res_x2 = minimize(
                obj_x2,
                x0,
                method="Powell",
                options={
                    "maxiter": 50,
                    "maxfev": 1500,
                    "ftol": 1e-16,
                    "xtol": 1e-16,
                    "disp": False,
                },
            )
            if res_x2.fun < best_vol:
                best_vol = res_x2.fun
                best_params[0:_N2] = res_x2.x
        except Exception:
            pass

        if time_left() > 15.0:
            x1 = best_params[0:_N2].copy()
            y1 = best_params[_N2:2 * _N2].copy()
            
            def obj_y2(y):
                return _exact_volume(np.concatenate([x1, y]))
            
            try:
                res_y2 = minimize(
                    obj_y2,
                    y1,
                    method="Powell",
                    options={
                        "maxiter": 50,
                        "maxfev": 1500,
                        "ftol": 1e-16,
                        "xtol": 1e-16,
                        "disp": False,
                    },
                )
                if res_y2.fun < best_vol:
                    best_vol = res_y2.fun
                    best_params[_N2:2 * _N2] = res_y2.x
            except Exception:
                pass

    x = best_params[0:_N2]
    y = best_params[_N2:2 * _N2]
    positions = np.stack([x, y], axis=1)
    return positions

# EVOLVE-BLOCK-END
