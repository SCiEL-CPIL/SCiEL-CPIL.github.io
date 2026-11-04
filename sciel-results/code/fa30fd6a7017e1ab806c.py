"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre

# Precompute factorials for moment computation
# We need k! for k up to 48 (degree of Q = P^2 where P is degree 24)
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (given in ascending coefficient order) * e^{-x} dx over [0,inf).
    coeffs[k] is coefficient of x^k. Result = sum_k coeffs[k] * k!."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    # Use dot product; factorial grows fast so for large degree this can overflow in float64
    # but we normalize by I1^2 so it should be OK if we scale
    return float(np.dot(coeffs[:min(n, 60)], _FACTS[:min(n, 60)]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs).
    Coefficients in ascending power order (np.polynomial.polynomial convention)."""
    # P(x) = prod (x - x_i)^2 = [prod (x - x_i)]^2
    # Build base polynomial B(x) = prod (x - x_i), then P = B^2
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)  # ascending order
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2. Returns (c4, grad) or just c4."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Gradient: dC4/dx_i = (dI2/dx_i * I1^2 - I2 * 2 * I1 * dI1/dx_i) / I1^4
    # = dI2/dx_i / I1^2 - 2 * I2 * dI1/dx_i / I1^3
    # dI1/dx_i: dP/dx_i where P = (x-x_i)^2 * R_i, R_i = prod_{j!=i}(x-x_j)^2
    # dP/dx_i = -2(x-x_i) * R_i
    # So dI1/dx_i = -2 * integral((x-x_i)*R_i * e^{-x} dx)
    # Similarly P2 = P^2, dP2/dx_i = 2*P*dP/dx_i = -4*(x-x_i)*R_i*P
    # dI2/dx_i = -4 * integral((x-x_i)*R_i*P * e^{-x} dx)
    
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        R_i = np.ones(1)
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = np.polynomial.polynomial.polymul(R_i, np.polynomial.polynomial.polyfromroots([roots[j], roots[j]]))
        # dI1: -2 * int (x-x_i)*R_i e^{-x} dx
        # (x-x_i)*R_i = x*R_i - x_i*R_i
        xR_i = np.polynomial.polynomial.polymul(R_i, np.array([0.0, 1.0]))  # multiply by x
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        # dI2: -4 * int (x-x_i)*R_i*P e^{-x} dx
        RP = np.polynomial.polynomial.polymul(R_i, P)
        xRP = np.polynomial.polynomial.polymul(RP, np.array([0.0, 1.0]))
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    c4, g = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.inf, np.zeros_like(roots)
    return c4, g

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0  # return 45s before kill
    deadline = start + DEADLINE - SAFETY

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)

    # Change of variables: roots = exp(t) to remove positivity constraint
    # This allows unconstrained optimization in log-space
    def _objective_log(t):
        roots = np.exp(t)
        return _compute_c4(roots)
    
    def _objective_log_grad(t):
        roots = np.exp(t)
        c4, grad_roots = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return np.inf, np.zeros_like(t)
        # Chain rule: dC4/dt_i = dC4/dx_i * dx_i/dt_i = dC4/dx_i * x_i
        grad_t = grad_roots * roots
        return c4, grad_t

    # Differential evolution in log-space for global search
    if time_ok():
        # t bounds: roots in [1e-4, 100] -> t in [log(1e-4), log(100)]
        t_bounds = [(np.log(1e-4), np.log(100.0))] * 12
        try:
            de_result = differential_evolution(
                _objective_log,
                bounds=t_bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=False,
                update_interval=5,
                callback=lambda x, convergence: None
            )
            if np.isfinite(de_result.fun):
                candidate = np.sort(np.exp(de_result.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # L-BFGS-B refinement in log-space from best point and several others
    candidates_to_refine_t = [np.log(best_roots.copy())]
    # Add perturbed versions in log-space
    rng = np.random.default_rng(123)
    for _ in range(6):
        pert_t = np.log(best_roots) + 0.2 * rng.standard_normal(12)
        candidates_to_refine_t.append(pert_t)

    for init_t in candidates_to_refine_t:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_log_grad,
                init_t,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final high-precision polish in log-space
    if time_ok():
        try:
            t_init = np.log(best_roots)
            res = minimize(
                _objective_log_grad,
                t_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 30}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Additional polish: Nelder-Mead in log-space for robustness
    if time_ok():
        try:
            t_init = np.log(best_roots)
            res = minimize(
                _objective_log,
                t_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # Final safety check
    if not time_ok():
        pass  # just return best found

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
