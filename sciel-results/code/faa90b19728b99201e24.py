"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

construct() -> list of 12 positive double-root locations (m=12, n=1).
The score is the upper bound on C4 (lower is better).
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import roots_laguerre, eval_laguerre, laguerre
from scipy.optimize import minimize, differential_evolution

_M = 12
_N = 1
_DEADLINE = 600.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

# ---- polynomial helpers (coefficients, high degree first) ----
def _poly_mul(a, b):
    return np.polymul(a, b)

def _poly_add(a, b):
    return np.polyadd(a, b)

def _poly_sub(a, b):
    return np.polysub(a, b)

def _poly_scalar(c, a):
    return c * a

def _build_poly(r):
    """P(x) = L_{2m}(x) + sum_{j} L_{2m}(x - r_j).
    Uses degree 2*m to allow for m double roots."""
    m = 2 * _M
    Lm = laguerre(m)
    P = Lm.c
    for rj in r:
        Lm_shift = laguerre(m, rj)
        P = _poly_add(P, Lm_shift.c)
    return P

def _roots(P):
    return np.roots(P)

def _validate(r, tol=1e-7):
    """Return (ok, P, roots) if r are positive, increasing double roots of P."""
    if len(r) != _M:
        return False, None, None
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)):
        return False, None, None
    if np.min(r) <= 0:
        return False, None, None
    if np.any(np.diff(r) <= 1e-9):
        return False, None, None
    P = _build_poly(r)
    roots = _roots(P)
    real = roots[np.abs(roots.imag) < 1e-5]
    if len(real) != 2 * _M:
        return False, None, None
    real = np.sort(real.real)
    for j in range(_M):
        lo, hi = real[2 * j], real[2 * j + 1]
        if lo <= 0:
            return False, None, None
        if abs(lo - hi) > tol or abs((lo + hi) / 2.0 - r[j]) > tol:
            return False, None, None
    return True, P, real

def _C4(r):
    """Upper bound on C4 = product over double roots of (r_j + 1/r_j)."""
    r = np.asarray(r, dtype=float)
    if np.min(r) <= 0:
        return float("inf")
    return float(np.prod(r + 1.0 / r))

def _penalized(r, P, roots, w_pos, w_double, w_order, w_real):
    """Smooth-ish penalty for constraint violations (used when hard check fails)."""
    r = np.asarray(r, dtype=float)
    pen = 0.0
    if np.min(r) <= 0:
        pen += w_pos * (1.0 - np.min(r)) ** 2
    if np.any(np.diff(r) <= 0):
        pen += w_order * np.sum(np.maximum(0.0, -np.diff(r))) ** 2
    real = roots[np.abs(roots.imag) < 1e-3]
    if len(real) < 2 * _M:
        pen += w_real * (2 * _M - len(real)) ** 2
    else:
        real = np.sort(real.real)
        for j in range(_M):
            lo, hi = real[2 * j], real[2 * j + 1]
            mid = (lo + hi) / 2.0
            pen += w_double * (abs(hi - lo) + abs(mid - r[j])) ** 2
    return pen

def _residuals(r):
    """Compute residuals for least_squares: P(r_j) and P'(r_j) should be 0."""
    r = np.asarray(r, dtype=float)
    P = _build_poly(r)
    P_prime = np.polyder(P)
    res = []
    for rj in r:
        val = np.polyval(P, rj)
        der = np.polyval(P_prime, rj)
        res.append(val)
        res.append(der)
    return np.array(res)

def _optimize_from(r0, budget):
    """Local polish: refine r to satisfy double-root constraints and minimize C4."""
    best_r = np.asarray(r0, dtype=float).copy()
    best_val = _C4(best_r)
    
    # Use least_squares to drive residuals to zero
    if _time_left() > 10:
        try:
            from scipy.optimize import least_squares
            res = least_squares(_residuals, best_r, method='lm', max_nfev=10000)
            cand = res.x
            if np.all(np.isfinite(cand)) and np.min(cand) > 0:
                # Sort and check validity
                cand = np.sort(cand)
                ok, _, _ = _validate(cand)
                if ok:
                    val = _C4(cand)
                    if val < best_val:
                        best_val = val
                        best_r = cand
        except Exception:
            pass
            
    # If still not valid or not improved, try minimize on penalized objective
    if not _validate(best_r)[0]:
        w_pos, w_double, w_order, w_real = 1e6, 1e6, 1e6, 1e6
        def objective(r):
            P = _build_poly(r)
            roots = _roots(P)
            ok, _, _ = _validate(r)
            if ok:
                return _C4(r)
            return _C4(r) + _penalized(r, P, roots, w_pos, w_double, w_order, w_real)
        for method in ("BFGS", "Nelder-Mead"):
            if _time_left() < 10:
                break
            try:
                res = minimize(objective, best_r, method=method,
                               options={"maxiter": 2000, "xatol": 1e-12, "fatol": 1e-12})
                if res.success and np.all(np.isfinite(res.x)):
                    cand = np.sort(res.x)
                    ok, _, _ = _validate(cand)
                    if ok and _C4(cand) < best_val:
                        best_val = _C4(cand)
                        best_r = cand
            except Exception:
                pass
                
    return best_r, best_val

def _heuristic_initial():
    """Generate a plausible starting set of 12 increasing positive locations."""
    # Start from the Laguerre roots region and spread out; use a log-spaced grid.
    base = np.logspace(math.log10(0.5), math.log10(50.0), _M)
    return base

def construct():
    # Deadline guard
    if _time_left() < 5:
        return [4.0 * (i + 1) for i in range(_M)]

    # 1) Global search for a feasible configuration
    best_r = None
    best_val = float("inf")

    # Candidate initial points
    inits = [_heuristic_initial()]
    # Also try the seed's linear spacing as one candidate
    inits.append(np.array([4.0 * (i + 1) for i in range(_M)], dtype=float))
    # Add a couple of random log-spaced grids
    rng = np.random.default_rng(12345)
    for _ in range(4):
        lo = rng.uniform(0.2, 2.0)
        hi = rng.uniform(20.0, 100.0)
        inits.append(np.logspace(math.log10(lo), math.log10(hi), _M))
    # Add linear grids with different scales
    for scale in [1.0, 2.0, 5.0]:
        inits.append(np.array([scale * (i + 1) for i in range(_M)], dtype=float))

    # 2) For each initial point, run local optimization and keep best feasible
    for r0 in inits:
        if _time_left() < 60:
            break
        r_opt, val = _optimize_from(r0, 60.0)
        ok, _, _ = _validate(r_opt)
        if ok and val < best_val:
            best_val = val
            best_r = r_opt

    # 3) If nothing feasible yet, try a broader differential evolution on the
    #    penalized objective to find a feasible region, then polish.
    if best_r is None and _time_left() > 120:
        def obj_de(r):
            P = _build_poly(r)
            roots = _roots(P)
            ok, _, _ = _validate(r)
            if ok:
                return _C4(r)
            return _C4(r) + _penalized(r, P, roots, 1e6, 1e6, 1e6, 1e6)
        bounds = [(0.1, 200.0)] * _M
        try:
            res = differential_evolution(obj_de, bounds, maxiter=200, tol=1e-9,
                                         seed=12345, polish=False, updating="deferred")
            if np.all(np.isfinite(res.x)):
                r_opt, val = _optimize_from(res.x, 90.0)
                ok, _, _ = _validate(r_opt)
                if ok and val < best_val:
                    best_val = val
                    best_r = r_opt
        except Exception:
            pass

    # 4) Final high-precision polish on the best candidate
    if best_r is not None and _time_left() > 30:
        r_opt, val = _optimize_from(best_r, 60.0)
        ok, _, _ = _validate(r_opt)
        if ok and val < best_val:
            best_val = val
            best_r = r_opt

    # 5) Return a valid, sorted list of 12 positive double-root locations
    if best_r is None or not np.all(np.isfinite(best_r)) or np.min(best_r) <= 0:
        # Fallback: return seed values (still a valid increasing positive list)
        return [4.0 * (i + 1) for i in range(_M)]

    best_r = np.sort(best_r)
    # Ensure strictly positive and increasing
    out = []
    prev = 0.0
    for v in best_r:
        v = float(v)
        if v <= prev:
            v = prev + 1e-6
        out.append(v)
        prev = v
    return out
# EVOLVE-BLOCK-END
