import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul
import time
import math

# Precompute factorials up to 80 (degree of P^2 is 48, but we need moments up to that)
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

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
    # P = B^2, dP/dx_i = 2*B * dB/dx_i = 2*B * (-1)*prod_{j!=i}(x-x_j) = -2*B*R_i
    # where R_i = prod_{j!=i}(x-x_j)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # P2 = B^4, dP2/dx_i = 4*B^3 * dB/dx_i = -4*B^3*R_i
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    
    grad = np.zeros_like(roots)
    # Precompute B^3 once
    B2 = polymul(B, B)
    B3 = polymul(B2, B)  # degree 36
    
    for i in range(n):
        # R_i = B / (x - x_i) = prod_{j!=i}(x-x_j)
        # Build R_i directly
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([1.0, -roots[j]]))  # (x - x_j) in ascending: [-x_j, 1]
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)  # degree 12 + 11 = 23
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)  # degree 36 + 11 = 47
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _c4_from_logspacings(log_ratios):
    """Compute C4 for roots generated from log-spaced ratios.
    roots[i] = x_0 * prod_{k=1}^{i} r_k, where r_k = exp(log_ratios[k-1])
    This parameterization ensures all roots are positive and ordered."""
    n = len(log_ratios)
    x0 = np.exp(log_ratios[0])
    ratios = np.exp(log_ratios[1:])
    roots = np.empty(n)
    roots[0] = x0
    for i in range(1, n):
        roots[i] = roots[i-1] * ratios[i-1]
    return _compute_c4(roots)

def _logspacings_from_roots(roots):
    """Convert roots to log-spacing parameterization."""
    n = len(roots)
    log_ratios = np.empty(n)
    log_ratios[0] = np.log(roots[0])
    for i in range(1, n):
        log_ratios[i] = np.log(roots[i] / roots[i-1])
    return log_ratios

def _logspacings_grad(roots):
    """Compute gradient of C4 w.r.t. log-spacing parameters."""
    c4, grad_roots = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    
    n = len(roots)
    grad_ls = np.zeros(n)
    # dC4/dlog_ratios[0] = dC4/dx0 * x0
    grad_ls[0] = grad_roots[0] * roots[0]
    # For i >= 1: log_ratios[i] affects roots[i], roots[i+1], ..., roots[n-1]
    # droots[k]/dlog_ratios[i] = roots[k] for k >= i
    for i in range(1, n):
        grad_ls[i] = np.sum(grad_roots[i:] * roots[i:])
    return c4, grad_ls

def _objective_ls(ls):
    """Objective in log-spacing space."""
    roots = _c4_from_logspacings_helper(ls)
    return _compute_c4(roots)

def _objective_ls_grad(ls):
    """Objective with gradient in log-spacing space."""
    roots = _c4_from_logspacings_helper(ls)
    c4, grad_roots = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(ls)
    
    n = len(roots)
    grad_ls = np.zeros(n)
    grad_ls[0] = grad_roots[0] * roots[0]
    for i in range(1, n):
        grad_ls[i] = np.sum(grad_roots[i:] * roots[i:])
    return c4, grad_ls

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

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    # Grid search over log-spaced alpha values
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 500)
    for la in log_alphas:
        if not time_ok():
            break
        c4_val = _c4_from_alpha(np.exp(la))
        if c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = np.exp(la)

    # Refine with scalar minimization
    if time_ok():
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(np.exp(la)),
                bounds=(math.log(max(best_alpha * 0.5, 1e-4)), math.log(min(best_alpha * 2.0, 1e5))),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 500}
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

    # Also try standard Laguerre (alpha=0)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()

    # Phase 2: Log-spacing parameterization optimization
    # Convert best roots to log-spacing parameters
    best_ls = _logspacings_from_roots(best_roots)
    
    # Optimize in log-space with L-BFGS-B using proper gradient
    if time_ok():
        try:
            res_ls = minimize(
                _objective_ls,
                best_ls,
                jac=_objective_ls_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
            )
            if np.isfinite(res_ls.fun):
                # Reconstruct roots from log-spacings
                roots_ls = _c4_from_logspacings_helper(res_ls.x)
                roots_ls = np.sort(roots_ls)
                c4_ls = _compute_c4(roots_ls)
                if c4_ls < best_c4:
                    best_c4 = c4_ls
                    best_roots = roots_ls.copy()
                    best_ls = _logspacings_from_roots(best_roots)
        except Exception:
            pass

    # Phase 3: Full 12-DOF L-BFGS-B refinement with analytic gradient (in original space)
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with perturbed initializations in log-space
    if time_ok():
        rng = np.random.default_rng(42)
        for trial in range(30):
            if not time_ok():
                break
            # Perturb in log-space
            pert_ls = best_ls + 0.3 * rng.standard_normal(12)
            try:
                res_pert = minimize(
                    _objective_ls,
                    pert_ls,
                    jac=_objective_ls_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 80}
                )
                if np.isfinite(res_pert.fun):
                    roots_pert = _c4_from_logspacings_helper(res_pert.x)
                    roots_pert = np.sort(roots_pert)
                    c4_pert = _compute_c4(roots_pert)
                    if c4_pert < best_c4:
                        best_c4 = c4_pert
                        best_roots = roots_pert.copy()
                        best_ls = _logspacings_from_roots(best_roots)
            except Exception:
                pass

    # Phase 5: High-precision polish pass
    if time_ok() and time.time() - start < DEADLINE - 60:
        try:
            # Use TNC for potentially better convergence
            res_tnc = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='TNC',
                bounds=[(1e-12, None)] * 12,
                options={'maxiter': 10000, 'ftol': 1e-22, 'gtol': 1e-20}
            )
            if np.all(res_tnc.x > 0) and np.isfinite(res_tnc.fun):
                candidate = np.sort(res_tnc.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final safety check
    if time.time() > DEADLINE - 5:
        return best_roots.tolist()
    
    return best_roots.tolist()

def _c4_from_logspacings_helper(ls):
    """Helper to reconstruct roots from log-spacings for gradient computation."""
    x0_val = np.exp(ls[0])
    ratios = np.exp(ls[1:])
    roots = np.empty(12)
    roots[0] = x0_val
    for i in range(1, 12):
        roots[i] = roots[i-1] * ratios[i-1]
    return roots
