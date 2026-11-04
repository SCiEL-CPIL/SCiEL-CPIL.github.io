"""Combined: analytic C4 with polynomial moments + global search via Laguerre basis + multi-start polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials up to 60 (degree of P^2 is 48)
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

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
    grad = np.zeros_like(roots)
    # Precompute B^3
    B2 = polymul(B, B)
    B3 = polymul(B2, B)
    
    for i in range(n):
        # R_i = prod_{j!=i}(x-x_j)
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([-roots[j], 1.0]))
        
        # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
        BR_i = polymul(B, R_i)
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        # dI2/dx_i = -4 * int B^3 * R_i * e^{-x} dx
        B3R_i = polymul(B3, R_i)
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    def time_ok():
        return (time.time() - start_time) < deadline - 30

    # Phase 1: Global search using differential evolution in log-space
    # The objective is C4 = I2/I1^2, which is smooth for positive roots
    def objective_log(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(roots > 1e6):
            return 1e10
        c4 = _compute_c4(roots)
        if not np.isfinite(c4):
            return 1e10
        return c4

    # Bounds: log(0.1) to log(1000) should cover reasonable root locations
    log_bounds = [(-2.3, 6.9)] * 12  # log(0.1) to log(1000)

    best_roots = None
    best_c4 = np.inf

    # Run differential evolution with multiple seeds
    de_seeds = [42, 123, 456]
    for seed in de_seeds:
        if not time_ok():
            break
        try:
            result = differential_evolution(
                objective_log,
                log_bounds,
                seed=seed,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            if np.isfinite(result.fun) and result.fun < best_c4:
                best_c4 = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass

    # Fallback: try standard Laguerre roots
    try:
        x0, _ = roots_laguerre(12)
        x0 = np.sort(x0)
        c4_0 = _compute_c4(x0)
        if np.isfinite(c4_0) and c4_0 < best_c4:
            best_c4 = c4_0
            best_roots = x0.copy()
    except Exception:
        pass

    # Fallback: geometric sequence
    if best_roots is None:
        best_roots = 4.0 * (2.0 ** np.arange(12))
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Full 12-DOF L-BFGS-B refinement with analytic gradient
    def objective(roots):
        c4 = _compute_c4(roots)
        return c4
    
    def objective_grad(roots):
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return 1e10, np.zeros_like(roots)
        return c4, g

    if time_ok() and best_roots is not None:
        try:
            res = minimize(
                objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, 1e4)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun) and res.fun < best_c4:
                best_c4 = res.fun
                best_roots = np.sort(res.x)
        except Exception:
            pass

    # Phase 3: Multi-start with perturbed initializations
    if time_ok():
        rng = np.random.default_rng(42)
        for trial in range(8):
            if not time_ok():
                break
            # Perturb the best roots
            pert = best_roots * (1.0 + 0.2 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-4)
            pert = np.sort(pert)
            try:
                res = minimize(
                    objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 1e4)] * 12,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun) and res.fun < best_c4:
                    best_c4 = res.fun
                    best_roots = np.sort(res.x)
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if time_ok():
        try:
            res_final = minimize(
                objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, 1e5)] * 12,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.all(res_final.x > 0) and np.isfinite(res_final.fun) and res_final.fun < best_c4:
                best_c4 = res_final.fun
                best_roots = np.sort(res_final.x)
        except Exception:
            pass

    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        # Fallback to standard Laguerre roots
        try:
            x0, _ = roots_laguerre(12)
            best_roots = np.sort(x0)
        except Exception:
            best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
