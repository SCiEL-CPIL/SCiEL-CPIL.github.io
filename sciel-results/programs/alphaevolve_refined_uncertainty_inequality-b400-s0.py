"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize.
Uses 1D search over generalized Laguerre alpha + multi-start L-BFGS-B with analytic gradient."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul, polydiv

# Precompute factorials for moment computation (degree of P^2 is 48, need up to 48)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, len(_FACTS))
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_q_and_p(roots):
    """Build Q = prod(x-x_i), P = Q^2, P2 = P^2. Return (Q, P, P2)."""
    Q = polyfromroots(roots)
    P = polymul(Q, Q)
    P2 = polymul(P, P)
    return Q, P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    Q, P, P2 = _build_q_and_p(roots)
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient:
    # Q = prod_j (x - x_j), P = Q^2, P2 = Q^4
    # dQ/dx_i = -Q/(x - x_i) = -R_i where R_i = prod_{j!=i}(x-x_j)
    # dP/dx_i = 2Q * dQ/dx_i = -2Q*R_i = -2P/(x-x_i)
    # dP2/dx_i = 4Q^3 * dQ/dx_i = -4Q^3*R_i = -4P2/(x-x_i)
    #
    # dI1/dx_i = -2 * int Q*R_i * e^{-x} dx
    # dI2/dx_i = -4 * int Q^3*R_i * e^{-x} dx
    #
    # Note: Q*R_i = Q * prod_{j!=i}(x-x_j) = (x-x_i) * [prod_{j!=i}(x-x_j)]^2
    # But more directly: Q*R_i = Q * (Q/(x-x_i)) = Q^2/(x-x_i) = P/(x-x_i)
    # Similarly: Q^3*R_i = Q^3 * Q/(x-x_i) = Q^4/(x-x_i) = P2/(x-x_i)
    #
    # So: dI1/dx_i = -2 * int P/(x-x_i) e^{-x} dx
    #     dI2/dx_i = -4 * int P2/(x-x_i) e^{-x} dx
    #
    # Use polynomial division: P/(x-x_i) and P2/(x-x_i)
    # Since P and P2 have (x-x_i)^2 as factor, the division is exact.
    
    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = np.zeros(n)
    
    for i in range(n):
        # Compute P/(x - roots[i]) via polydiv
        divisor = np.array([-roots[i], 1.0])
        
        # P/(x - x_i): P has degree 24, result has degree 23
        S_i, _ = polydiv(P, divisor)
        int_S = _poly_moment(S_i)
        dI1_i = -2.0 * int_S
        
        # P2/(x - x_i): P2 has degree 48, result has degree 47
        T_i, _ = polydiv(P2, divisor)
        int_T = _poly_moment(T_i)
        dI2_i = -4.0 * int_T
        
        grad[i] = dI2_i / I1_sq - 2.0 * I2 * dI1_i / I1_cu
    
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
    DEADLINE = 620.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE

    best_roots = None
    best_c4 = np.inf
    best_alpha = 0.0

    # Phase 1: 1D search over alpha for generalized Laguerre nodes
    # Coarse log-spaced grid
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 200)
    for la in log_alphas:
        if not time_ok():
            break
        alpha = np.exp(la)
        c4_val = _c4_from_alpha(alpha)
        if np.isfinite(c4_val) and c4_val < best_c4:
            best_c4 = c4_val
            best_alpha = alpha
            xg, _ = roots_genlaguerre(12, alpha)
            best_roots = np.sort(xg)

    # Also try standard Laguerre (alpha=0)
    try:
        x0, _ = roots_laguerre(12)
        x0 = np.sort(x0)
        c4_0 = _compute_c4(x0)
        if np.isfinite(c4_0) and c4_0 < best_c4:
            best_c4 = c4_0
            best_roots = x0.copy()
            best_alpha = 0.0
    except Exception:
        pass

    # Fallback
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)
        best_alpha = 0.0

    # Phase 2: Refine alpha with scalar minimization
    if time_ok() and best_alpha > 0:
        try:
            a_lo = max(best_alpha * 0.5, 1e-4)
            a_hi = best_alpha * 2.0
            res_alpha = minimize_scalar(
                _c4_from_alpha,
                bounds=(a_lo, a_hi),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 200}
            )
            if np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4:
                best_alpha = res_alpha.x
                xg, _ = roots_genlaguerre(12, best_alpha)
                best_roots = np.sort(xg)
                best_c4 = res_alpha.fun
        except Exception:
            pass
        # Second refinement
        if time_ok():
            try:
                a_lo = max(best_alpha * 0.8, 1e-4)
                a_hi = best_alpha * 1.2
                res_alpha2 = minimize_scalar(
                    _c4_from_alpha,
                    bounds=(a_lo, a_hi),
                    method='bounded',
                    options={'xatol': 1e-12, 'maxiter': 200}
                )
                if np.isfinite(res_alpha2.fun) and res_alpha2.fun < best_c4:
                    best_alpha = res_alpha2.x
                    xg, _ = roots_genlaguerre(12, best_alpha)
                    best_roots = np.sort(xg)
                    best_c4 = res_alpha2.fun
            except Exception:
                pass

    # Phase 3: Full 12-DOF L-BFGS-B refinement from best alpha point
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots.copy(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start L-BFGS-B from perturbed best and neighboring alphas
    if time_ok():
        rng = np.random.default_rng(42)
        candidates_to_refine = [best_roots.copy()]
        
        # Perturbations around best
        for scale in [0.02, 0.05, 0.1, 0.2, 0.5]:
            for _ in range(2):
                if not time_ok():
                    break
                pert = best_roots * (1.0 + scale * rng.standard_normal(12))
                pert = np.maximum(pert, 1e-3)
                pert = np.sort(pert)
                candidates_to_refine.append(pert)
        
        # Neighboring alpha values
        if best_alpha > 0:
            for factor in [0.5, 0.7, 1.3, 2.0, 0.3, 3.0]:
                if not time_ok():
                    break
                a_try = best_alpha * factor
                if a_try < 1e-4 or a_try > 1e5:
                    continue
                try:
                    xg2, _ = roots_genlaguerre(12, a_try)
                    xg2 = np.sort(xg2)
                    if np.all(xg2 > 0):
                        candidates_to_refine.append(xg2)
                except Exception:
                    pass
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_grad,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish passes
    for polish_iter in range(8):
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 60}
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

    # Final safety check
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
