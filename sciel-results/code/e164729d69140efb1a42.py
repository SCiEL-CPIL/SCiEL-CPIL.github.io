"""Model-and-optimize: parameterize 12 double roots, compute C4 bound via polynomial moments, optimize."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, root
from scipy.special import roots_laguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:min(n, 60)], _FACTS[:min(n, 60)]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
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
    
    grad = np.zeros_like(roots)
    from numpy.polynomial.polynomial import polymul as _pmul, polyfromroots as _pfr
    for i in range(len(roots)):
        R_i = np.ones(1)
        for j in range(len(roots)):
            if j == i:
                continue
            R_i = _pmul(R_i, _pfr([roots[j], roots[j]]))
        xR_i = _pmul(R_i, np.array([0.0, 1.0]))
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        RP = _pmul(R_i, P)
        xRP = _pmul(RP, np.array([0.0, 1.0]))
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

def _equilibrium_equations(log_roots):
    """Solve the stationarity conditions in log-space.
    The gradient of C4 w.r.t. each root should be zero at optimum.
    We parameterize as log(r_i) to ensure positivity."""
    roots = np.exp(log_roots)
    if np.any(roots > 1e8) or np.any(roots < 1e-8):
        return np.ones(12) * 1e6
    c4, grad = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return np.ones(12) * 1e6
    # Scale gradient by roots to work in log-space: dC4/d(log r_i) = r_i * dC4/dr_i
    return roots * grad

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 45.0

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

    # Phase 1: Solve equilibrium equations in log-space with multiple initial guesses
    # This is a structural change: instead of generic minimization, solve F(log_roots)=0
    # where F_i = r_i * dC4/dr_i = 0
    if time_ok():
        rng = np.random.default_rng(42)
        # Generate diverse initial guesses in log-space
        initial_guesses = [np.log(x0.copy())]
        
        # Geometric progression guesses
        for b in [1.2, 1.5, 2.0, 2.5, 3.0]:
            for a in [0.1, 0.5, 1.0, 2.0, 5.0]:
                roots = a * (b ** np.arange(12))
                if np.all(roots > 0) and np.all(roots < 1e5):
                    initial_guesses.append(np.log(roots))
        
        # Arithmetic progression guesses
        for d in [0.5, 1.0, 2.0, 3.0]:
            for a in [0.1, 0.5, 1.0]:
                roots = a + np.arange(12) * d
                if np.all(roots > 0) and np.all(roots < 1e5):
                    initial_guesses.append(np.log(roots))
        
        # Perturbed Laguerre
        for _ in range(10):
            pert = x0 * (1.0 + 0.3 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-4)
            pert = np.sort(pert)
            initial_guesses.append(np.log(pert))
        
        # Try solving from each initial guess
        for guess in initial_guesses:
            if not time_ok():
                break
            try:
                sol = root(_equilibrium_equations, guess, method='hybr', tol=1e-10, maxfev=5000)
                if sol.success or np.all(np.abs(sol.fun) < 1e-6):
                    candidate_roots = np.exp(sol.x)
                    if np.all(candidate_roots > 0) and np.all(np.isfinite(candidate_roots)):
                        candidate_roots = np.sort(candidate_roots)
                        c4 = _compute_c4(candidate_roots)
                        if np.isfinite(c4):
                            update_best(candidate_roots, c4)
            except Exception:
                pass

    # Phase 2: Differential evolution in log-space for global search
    if time_ok():
        def _log_obj(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots > 1e5) or np.any(roots < 1e-6):
                return np.inf
            return _compute_c4(roots)
        
        # Bounds in log-space: roots from 1e-4 to 1e4
        bounds = [(np.log(1e-4), np.log(1e4))] * 12
        try:
            de_result = differential_evolution(
                _log_obj,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                polish=False,
                update_interval=10
            )
            if np.isfinite(de_result.fun):
                candidate_roots = np.sort(np.exp(de_result.x))
                if np.all(candidate_roots > 0):
                    c4 = _compute_c4(candidate_roots)
                    update_best(candidate_roots, c4)
        except Exception:
            pass

    # Phase 3: Local refinement with L-BFGS-B in original space
    candidates_to_refine = [best_roots.copy()]
    if not np.allclose(best_roots, x0):
        candidates_to_refine.append(x0.copy())
    
    rng = np.random.default_rng(123)
    for _ in range(15):
        pert = best_roots * (1.0 + 0.15 * rng.standard_normal(12))
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

    # Phase 4: Final polish
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
