"""Combined: model-and-optimize with efficient analytic gradient + alpha-based init + multi-start polish.
Strategy: 1) Thorough 1D search over generalized Laguerre alpha parameter, 2) targeted multi-start L-BFGS-B, 3) high-precision polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials up to 80 (degree of P^2 is 48, but we need moments up to that)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _build_R_i(roots, i):
    """Build R_i = prod_{j!=i}(x - x_j) efficiently."""
    n = len(roots)
    R_i = np.array([1.0])
    for j in range(n):
        if j == i:
            continue
        R_i = polymul(R_i, np.array([-roots[j], 1.0]))
    return R_i

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient
    # P = B^2, dP/dx_i = 2*B * dB/dx_i = -2*B*R_i where R_i = prod_{j!=i}(x-x_j)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # P2 = B^4, dP2/dx_i = 4*B^3 * dB/dx_i = -4*B^3*R_i
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    
    grad = np.zeros_like(roots)
    # Precompute B^3 to avoid recomputing
    B3 = polymul(polymul(B, B), B)  # degree 36
    
    for i in range(n):
        # R_i = prod_{j!=i}(x-x_j)
        R_i = _build_R_i(roots, i)
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)  # degree 12 + 11 = 23
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)  # degree 36 + 11 = 47
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
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

def _refine_from_roots(roots, maxiter=2000, ftol=1e-16, gtol=1e-14):
    """Refine roots using L-BFGS-B and return (c4, refined_roots) if improved."""
    try:
        res = minimize(
            _objective_grad,
            roots,
            jac=True,
            method='L-BFGS-B',
            bounds=[(1e-10, None)] * len(roots),
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if np.all(res.x > 0) and np.isfinite(res.fun):
            candidate = np.sort(res.x)
            c4 = _compute_c4(candidate)
            return c4, candidate
    except Exception:
        pass
    return np.inf, None

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Phase 1: Thorough 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    # Coarse grid search over log-spaced alphas
    log_alphas = np.linspace(math.log(1e-3), math.log(1e5), 400)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Multi-scale refinement with scalar minimization
    for scale in [10.0, 5.0, 2.0, 1.0, 0.5]:
        if not time_ok():
            break
        lo = math.log(max(best_alpha / scale, 1e-4))
        hi = math.log(min(best_alpha * scale, 1e6))
        if lo >= hi:
            continue
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 1000}
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

    # Also try standard Laguerre (alpha=0) and a few nearby alphas
    for alpha_try in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if not time_ok():
            break
        try:
            xg_try, _ = roots_genlaguerre(12, alpha_try)
            xg_try = np.sort(xg_try)
            if not np.all(xg_try > 0):
                continue
            c4_try = _compute_c4(xg_try)
            if c4_try < best_c4:
                best_c4 = c4_try
                best_roots = xg_try.copy()
        except Exception:
            pass

    # Phase 2: Full 12-DOF L-BFGS-B refinement from alpha-optimal point
    if time_ok():
        c4_refined, roots_refined = _refine_from_roots(best_roots, maxiter=3000, ftol=1e-17, gtol=1e-15)
        if c4_refined < best_c4:
            best_c4 = c4_refined
            best_roots = roots_refined

    # Phase 3: Targeted multi-start with small perturbations around best
    if time_ok():
        rng = np.random.default_rng(42)
        for scale in [0.05, 0.1, 0.15, 0.2]:
            if not time_ok():
                break
            for trial in range(3):
                if not time_ok():
                    break
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                c4_pert, roots_pert = _refine_from_roots(pert, maxiter=2000, ftol=1e-16, gtol=1e-14)
                if c4_pert < best_c4:
                    best_c4 = c4_pert
                    best_roots = roots_pert

    # Phase 4: Try neighboring alpha values and refine
    if time_ok():
        for alpha_offset in [0.5, 2.0, 0.25, 4.0, 0.1, 10.0, 0.7, 1.4]:
            if not time_ok():
                break
            for sign in [-1, 1]:
                a_try = best_alpha * alpha_offset if sign > 0 else best_alpha / alpha_offset
                if a_try < 1e-3 or a_try > 1e5:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if not np.all(xg2 > 0):
                        continue
                    c4_2, roots_2 = _refine_from_roots(xg2, maxiter=2000, ftol=1e-16, gtol=1e-14)
                    if c4_2 < best_c4:
                        best_c4 = c4_2
                        best_roots = roots_2
                except Exception:
                    pass

    # Phase 5: Final high-precision polish: multiple passes with tightening tolerances
    for polish_iter in range(5):
        if not time_ok():
            break
        c4_polish, roots_polish = _refine_from_roots(
            best_roots, 
            maxiter=5000, 
            ftol=1e-18, 
            gtol=1e-16
        )
        if c4_polish < best_c4:
            best_c4 = c4_polish
            best_roots = roots_polish
        else:
            break  # No improvement, stop polishing

    # Final safety check and return
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        # Fallback to standard Laguerre roots if something went wrong
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
