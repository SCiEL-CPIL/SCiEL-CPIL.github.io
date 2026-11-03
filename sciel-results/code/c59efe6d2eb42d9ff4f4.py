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

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    def update_best(candidate_roots, c4_val):
        nonlocal best_roots, best_c4
        if c4_val < best_c4 and np.all(candidate_roots > 0) and np.all(np.isfinite(candidate_roots)):
            best_roots = np.sort(candidate_roots)
            best_c4 = c4_val

    # Initial guess: standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    best_roots = x0.copy()
    best_c4 = _compute_c4(best_roots)

    # 1. Structured Search: Geometric Progression
    # Hypothesis: Optimal roots follow x_i = a * b^i
    # Parameters: log_a, log_b
    def _geo_obj(params):
        log_a, log_b = params
        a = np.exp(log_a)
        b = np.exp(log_b)
        # Ensure b > 1 to keep roots sorted and positive, or handle sorting
        # If b < 1, roots are decreasing, we can sort them later
        # But for optimization, let's constrain b > 1.01
        if log_b < np.log(1.01):
            return np.inf
        roots = a * (b ** np.arange(12))
        # Check if roots are reasonable (not too large/small)
        if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(np.isnan(roots)):
            return np.inf
        return _compute_c4(roots)

    if time_ok():
        # Multistart for geometric
        rng = np.random.default_rng(42)
        for _ in range(20):
            # Start near Laguerre roots' geometric mean ratio
            # Laguerre roots are roughly linear in i for large n, but for n=12 they are spread out
            # Let's try a range of b
            init_log_b = rng.uniform(np.log(1.1), np.log(3.0))
            # Estimate a from the mean of Laguerre roots
            init_log_a = np.log(np.mean(x0)) + rng.normal(0, 0.1)
            init_params = [init_log_a, init_log_b]
            try:
                res = minimize(_geo_obj, init_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_c4:
                    log_a, log_b = res.x
                    a = np.exp(log_a)
                    b = np.exp(log_b)
                    candidate = a * (b ** np.arange(12))
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    update_best(candidate, c4)
            except Exception:
                pass

    # 2. Structured Search: Arithmetic Progression
    # Hypothesis: Optimal roots follow x_i = a + i * d
    def _arith_obj(params):
        a, d = params
        if d <= 0: return np.inf
        roots = a + np.arange(12) * d
        if np.any(roots <= 0) or np.any(roots > 1e6):
            return np.inf
        return _compute_c4(roots)
        
    if time_ok():
        rng = np.random.default_rng(43)
        for _ in range(20):
            init_a = rng.uniform(0.1, 5.0)
            init_d = rng.uniform(0.1, 2.0)
            init_params = [init_a, init_d]
            try:
                res = minimize(_arith_obj, init_params, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_c4:
                    a, d = res.x
                    candidate = a + np.arange(12) * d
                    candidate = np.sort(candidate)
                    c4 = _compute_c4(candidate)
                    update_best(candidate, c4)
            except Exception:
                pass

    # 3. Differential Evolution for Global Search
    # Use best found so far to guide bounds if possible, or standard bounds
    if time_ok():
        # Widen bounds slightly to capture outliers
        bounds = [(1e-3, 200.0)] * 12
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=300,
                popsize=20,
                tol=1e-10,
                polish=False,
                update_interval=10
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                update_best(candidate, c4)
        except Exception:
            pass

    # 4. Local Refinement with L-BFGS-B
    # Gather candidates: Laguerre, Best Structured, Best DE, Perturbed Best
    candidates_to_refine = [best_roots.copy()]
    
    # Add Laguerre again if it wasn't the best
    if not np.allclose(best_roots, x0):
        candidates_to_refine.append(x0.copy())
        
    rng = np.random.default_rng(123)
    for _ in range(10):
        pert = best_roots * (1.0 + 0.1 * rng.standard_normal(12))
        pert = np.maximum(pert, 1e-4)
        pert = np.sort(pert)
        candidates_to_refine.append(pert)

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                update_best(candidate, c4)
        except Exception:
            pass

    # 5. Final High-Precision Polish
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                update_best(candidate, c4)
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
