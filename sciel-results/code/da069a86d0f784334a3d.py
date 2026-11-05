"""Combined: direct root-space optimization with efficient analytic gradient
+ generalized-Laguerre alpha-scan for multi-start initializations.
Avoids the Jacobi eigenvalue round-trip; optimizes the 12 roots directly
with a tight, fast analytic gradient. Uses log-space parameterization for
positivity and a structured multi-start over alpha (generalized Laguerre
parameter) to escape local minima."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import roots_laguerre, roots_genlaguerre, gammaln
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute log-factorials up to 60 (degree of P^2 is 48, need moments up to 48)
_LOG_FACTS = np.array([gammaln(k + 1) for k in range(60)], dtype=np.float64)
_FACTS = np.exp(_LOG_FACTS)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx over [0,inf).
    = sum_k coeffs[k] * k!
    Uses log-space for stability."""
    n = min(len(coeffs), len(_LOG_FACTS))
    if n == 0:
        return 0.0
    # Handle sign
    signs = np.sign(coeffs[:n])
    log_abs = np.log(np.abs(coeffs[:n])) + _LOG_FACTS[:n]
    # Sum with sign (watch for cancellation; for our use case values are positive)
    result = 0.0
    for i in range(n):
        if coeffs[i] == 0:
            continue
        if coeffs[i] > 0:
            result += math.exp(log_abs[i])
        else:
            result -= math.exp(log_abs[i])
    return float(result)

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2 = Q^2 where Q = prod (x-x_i).
    Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    Q = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(Q, Q)         # degree 24
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
    # dP/dx_i = 2*P/(x-x_i)  [since P = Q^2, dQ/dx_i = Q/(x-x_i), dP/dx_i = 2*Q*dQ/dx_i = 2*Q^2/(x-x_i) = 2P/(x-x_i)]
    # dP2/dx_i = 4*P2/(x-x_i)
    # dI1/dx_i = 2 * int [P/(x-x_i)] e^{-x} dx
    # dI2/dx_i = 4 * int [P2/(x-x_i)] e^{-x} dx
    # dC4/dx_i = (dI2 * I1^2 - I2 * 2*I1*dI1) / I1^4 = dI2/I1^2 - 2*I2*dI1/I1^3

    from numpy.polynomial.polynomial import polydiv

    dI1 = np.zeros(n)
    dI2 = np.zeros(n)

    for i in range(n):
        divisor = np.array([-roots[i], 1.0])
        # S_i = P / (x - roots[i])
        S_i, rem = polydiv(P, divisor)
        # T_i = P2 / (x - roots[i])
        T_i, rem2 = polydiv(P2, divisor)

        int_S = _poly_moment(S_i)
        int_T = _poly_moment(T_i)

        dI1[i] = 2.0 * int_S
        dI2[i] = 4.0 * int_T

    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu

    return float(c4), grad

def _objective_log(log_roots, need_grad=True):
    """Objective in log-space for positivity constraint.
    log_roots: array of 12 log-values.
    Returns (value, grad_log) where grad_log = grad_roots / roots."""
    try:
        roots = np.exp(log_roots)
        c4, grad_roots = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return 1e10, np.zeros_like(log_roots)
        grad_log = grad_roots / roots
        if not np.all(np.isfinite(grad_log)):
            return 1e10, np.zeros_like(log_roots)
        return c4, grad_log
    except Exception:
        return 1e10, np.zeros_like(log_roots)

def _objective_log_safe(log_roots):
    try:
        c4, g = _objective_log(log_roots, need_grad=True)
        if not np.isfinite(c4) or not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(log_roots)
        return c4, g
    except Exception:
        return 1e10, np.zeros_like(log_roots)

def _c4_from_alpha(alpha):
    """Compute C4 for roots of generalized Laguerre polynomial L_12^{(alpha)}."""
    try:
        xg, _ = roots_genlaguerre(12, alpha)
        xg = np.sort(xg)
        if not np.all(xg > 0) or not np.isfinite(xg).all():
            return np.inf, None
        return _compute_c4(xg), xg
    except Exception:
        return np.inf, None

def construct():
    start = time.time()
    DEADLINE = 620.0
    SAFETY = 10.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf

    # Phase 1: 1D scan over alpha (generalized Laguerre parameter)
    # This gives a family of 12-root configurations with different density profiles.
    best_alpha = 0.0
    best_c4_alpha = np.inf

    # Log-spaced scan
    log_alphas = np.concatenate([
        np.linspace(math.log(1e-4), math.log(1e4), 200),
        np.linspace(math.log(0.01), math.log(100), 100),
    ])
    log_alphas = np.unique(np.sort(log_alphas))

    for la in log_alphas:
        if not time_ok():
            break
        alpha = math.exp(la)
        c4_val, roots_val = _c4_from_alpha(alpha)
        if np.isfinite(c4_val) and c4_val < best_c4_alpha:
            best_c4_alpha = c4_val
            best_alpha = alpha
            best_roots = roots_val.copy()
            best_c4 = c4_val

    # Also try standard Laguerre (alpha=0)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    c4_0 = _compute_c4(x0)
    if np.isfinite(c4_0) and c4_0 < best_c4:
        best_c4 = c4_0
        best_roots = x0.copy()
        best_alpha = 0.0

    # Refine alpha with scalar minimization
    if time_ok() and best_alpha > 0:
        try:
            res_alpha = minimize_scalar(
                lambda la: _c4_from_alpha(math.exp(la))[0],
                bounds=(math.log(max(best_alpha * 0.5, 1e-5)), math.log(min(best_alpha * 2.0, 1e5))),
                method='bounded',
                options={'xatol': 1e-12, 'maxiter': 200}
            )
            if res_alpha.success and np.isfinite(res_alpha.fun) and res_alpha.fun < best_c4:
                best_alpha = math.exp(res_alpha.x)
                _, roots_ref = _c4_from_alpha(best_alpha)
                if roots_ref is not None:
                    best_c4 = res_alpha.fun
                    best_roots = np.sort(roots_ref)
        except Exception:
            pass

    # Fallback
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Direct L-BFGS-B optimization in log-root space
    if time_ok():
        try:
            init_log = np.log(best_roots)
            res = minimize(
                _objective_log_safe,
                init_log,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.all(np.isfinite(res.x)):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0) and np.isfinite(candidate).all():
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Multi-start from different alpha values and perturbations
    if time_ok():
        # Collect initializations
        inits = []
        inits.append(np.log(best_roots))

        # Different alpha values
        for alpha_try in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            try:
                _, roots_try = _c4_from_alpha(alpha_try)
                if roots_try is not None and np.all(roots_try > 0):
                    inits.append(np.log(np.sort(roots_try)))
            except Exception:
                pass

        # Perturbations of best
        rng = np.random.default_rng(123)
        for scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
            pert_log = np.log(best_roots) + scale * rng.standard_normal(12)
            inits.append(pert_log)

        for init_log in inits:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_log_safe,
                    init_log,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 80}
                )
                if np.all(np.isfinite(res.x)):
                    candidate = np.sort(np.exp(res.x))
                    if np.all(candidate > 0) and np.isfinite(candidate).all():
                        c4 = _compute_c4(candidate)
                        if np.isfinite(c4) and c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 4: High-precision polish iterations
    for polish_iter in range(15):
        if not time_ok():
            break
        try:
            init_log = np.log(best_roots)
            res = minimize(
                _objective_log_safe,
                init_log,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 200}
            )
            if np.all(np.isfinite(res.x)):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0) and np.isfinite(candidate).all():
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
        except Exception:
            break

    # Final safety check
    if best_roots is None or not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
