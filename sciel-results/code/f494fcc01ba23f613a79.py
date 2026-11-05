"""Refined C4 optimization: multi-start over generalized Laguerre alpha, 12-DOF L-BFGS-B with analytic gradient."""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build P(x) = prod (x-x_i)^2
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Compute gradient using prefix/suffix products
    n = len(roots)
    # prefix[i] = prod_{j=0}^{i-1} (x-x_j)^2
    # suffix[i] = prod_{j=i}^{n-1} (x-x_j)^2
    prefix = [np.ones(1)] * n
    suffix = [np.ones(1)] * n
    
    for j in range(n):
        if j > 0:
            prefix[j] = polymul(prefix[j-1], np.array([roots[j]**2, -2*roots[j], 1.0]))
        else:
            prefix[j] = np.ones(1)
    
    for j in range(n-1, -1, -1):
        if j < n-1:
            suffix[j] = polymul(suffix[j+1], np.array([roots[j]**2, -2*roots[j], 1.0]))
        else:
            suffix[j] = np.ones(1)
    
    grad = np.zeros(n)
    for i in range(n):
        # Q_i = product of all (x-x_j)^2 for j != i
        if i == 0:
            Q_i = suffix[1].copy()
        elif i == n-1:
            Q_i = prefix[n-1].copy()
        else:
            Q_i = polymul(prefix[i], suffix[i+1])
        
        # int Q_i e^{-x} dx
        int_Qi = _poly_moment(Q_i)
        # int x*Q_i e^{-x} dx
        xQi = np.polynomial.polynomial.polymul(Q_i, np.array([0.0, 1.0]))
        int_xQi = _poly_moment(xQi)
        
        # int P/(x-x_i) e^{-x} dx = int_xQi - x_i * int_Qi
        int_P_over_xi = int_xQi - roots[i] * int_Qi
        
        # dI1/dx_i = 2 * int (x-x_i) Q_i e^{-x} dx = 2 * int_P_over_xi
        dI1_i = 2.0 * int_P_over_xi
        
        # dI2/dx_i = 4 * int (x-x_i)^3 * Q_i^2 e^{-x} dx
        S_i = polymul(Q_i, Q_i)
        x3Si = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 0.0, 0.0, 1.0]))
        x2Si = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 0.0, 1.0]))
        xSi = np.polynomial.polynomial.polymul(S_i, np.array([0.0, 1.0]))
        
        int_x3Si = _poly_moment(x3Si)
        int_x2Si = _poly_moment(x2Si)
        int_xSi = _poly_moment(xSi)
        int_Si = _poly_moment(S_i)
        
        dI2_i = 4.0 * (int_x3Si - 3.0*roots[i]*int_x2Si + 3.0*roots[i]**2*int_xSi - roots[i]**3*int_Si)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf
        return _compute_c4(xg)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Coarse-to-fine 1D search over alpha
    def _log_alpha_objective(log_a):
        alpha = np.exp(log_a)
        if alpha < 1e-3 or alpha > 1e4:
            return np.inf
        return _c4_from_alpha(alpha)

    best_alpha = 0.0
    best_c4_alpha = np.inf
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 50)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _log_alpha_objective(la)
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Fine grid around best
    if time_ok():
        local_lo = math.log(max(best_alpha * 0.5, 1e-3))
        local_hi = math.log(min(best_alpha * 2.0, 1e4))
        if local_hi > local_lo:
            log_alphas_fine = np.linspace(local_lo, local_hi, 200)
            for la in log_alphas_fine:
                if not time_ok():
                    break
                c4_val = _log_alpha_objective(la)
                if c4_val < best_c4_alpha:
                    best_c4_alpha = c4_val
                    best_alpha = np.exp(la)

    # Scalar minimization for final alpha precision
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                _log_alpha_objective,
                bounds=(math.log(1e-3), math.log(1e4)),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 300}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4_alpha:
                best_alpha = np.exp(res_alpha.x)
                best_c4_alpha = res_alpha.fun
        except Exception:
            pass

    # Get roots from best alpha
    xg, _ = roots_genlaguerre(12, best_alpha)
    best_roots = np.sort(xg)
    best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Try multiple neighboring alpha values as additional starting points
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.1, 10.0, 0.2, 5.0]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e4:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    res2 = minimize(
                        _objective_grad,
                        xg2,
                        jac=True,
                        method='L-BFGS-B',
                        bounds=[(1e-8, None)] * 12,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                    )
                    if np.all(res2.x > 0) and np.isfinite(res2.fun):
                        candidate = np.sort(res2.x)
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
                except Exception:
                    pass

    # Phase 4: Final high-precision polish (multiple passes)
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 30}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
                else:
                    break
        except Exception:
            break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
