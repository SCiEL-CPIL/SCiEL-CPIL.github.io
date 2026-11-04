"""Combined: analytic C4 with polynomial moments + tridiagonal manifold + multi-start polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
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

def _tridiag_to_roots(params):
    """Convert tridiagonal matrix parameters to sorted eigenvalues.
    params: [d1, d2, ..., d12, b1, b2, ..., b11]
    Returns sorted eigenvalues (roots)."""
    n = 12
    d = params[:n]
    b = params[n:2*n-1]
    # Build tridiagonal matrix
    T = np.zeros((n, n))
    T[np.diag_indices(n)] = d
    for i in range(n-1):
        T[i, i+1] = b[i]
        T[i+1, i] = b[i]
    # Ensure symmetry and positive definiteness by shifting
    # Eigenvalues of tridiagonal
    try:
        eigvals = np.linalg.eigvalsh(T)
        eigvals = np.sort(eigvals)
        # Shift to ensure positive
        min_val = eigvals[0]
        if min_val <= 0.1:
            eigvals = eigvals - min_val + 0.5
        return eigvals
    except Exception:
        return None

def _roots_to_tridiag(roots):
    """Convert sorted positive roots to tridiagonal matrix parameters.
    Uses the fact that for a tridiagonal matrix, we can reconstruct from eigenvalues
    using the inverse Stieltjes procedure or simply use a simple mapping."""
    n = 12
    roots = np.asarray(roots, dtype=np.float64)
    # Simple parameterization: use roots directly as diagonal, 
    # and set off-diagonals based on spacing
    d = roots.copy()
    b = np.zeros(n-1)
    for i in range(n-1):
        # Off-diagonal related to spacing
        spacing = roots[i+1] - roots[i]
        b[i] = 0.5 * spacing
    return np.concatenate([d, b])

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    def time_ok():
        return (time.time() - start_time) < deadline - 30

    # Phase 1: 1D search over alpha to find optimal root density profile
    best_alpha = 0.0
    best_c4_alpha = np.inf
    
    # Grid search over log alpha
    log_alphas = np.linspace(math.log(1e-3), math.log(1e4), 200)
    for la in log_alphas:
        if not time_ok():
            break
        try:
            alpha_val = np.exp(la)
            xg, _ = roots_genlaguerre(12, alpha_val)
            xg = np.sort(xg)
            if not np.all(xg > 0) or not np.isfinite(xg).all():
                continue
            c4_val = _compute_c4(xg)
            if c4_val < best_c4_alpha:
                best_c4_alpha = c4_val
                best_alpha = alpha_val
        except Exception:
            continue

    # Refine with scalar minimization
    if time_ok() and best_alpha > 0:
        try:
            res_alpha = minimize_scalar(
                lambda la: _compute_c4(np.sort(roots_genlaguerre(12, np.exp(la))[0])),
                bounds=(math.log(max(best_alpha * 0.5, 1e-4)), math.log(min(best_alpha * 2.0, 1e5))),
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

    # Also try standard Laguerre (alpha=0)
    try:
        x0, _ = roots_laguerre(12)
        x0 = np.sort(x0)
        c4_0 = _compute_c4(x0)
        if c4_0 < best_c4:
            best_c4 = c4_0
            best_roots = x0.copy()
    except Exception:
        pass

    # Phase 2: Full 12-DOF L-BFGS-B refinement
    def objective(roots):
        c4 = _compute_c4(roots)
        return c4
    
    def objective_grad(roots):
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return np.inf, np.zeros_like(roots)
        return c4, g

    if time_ok():
        try:
            res = minimize(
                objective_grad,
                best_roots,
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

    # Phase 3: Tridiagonal manifold optimization
    if time_ok():
        try:
            # Initialize tridiagonal parameters from best roots
            tri_params = _roots_to_tridiag(best_roots)
            
            def tri_objective(params):
                roots = _tridiag_to_roots(params)
                if roots is None:
                    return np.inf
                c4 = _compute_c4(roots)
                return c4
            
            def tri_objective_grad(params):
                roots = _tridiag_to_roots(params)
                if roots is None:
                    return np.inf, np.zeros_like(params)
                c4, grad = _compute_c4(roots, need_grad=True)
                if not np.isfinite(c4):
                    return np.inf, np.zeros_like(params)
                # Need to compute gradient w.r.t. tridiagonal params
                # This is complex, so use numerical gradient or just objective
                return c4, np.zeros_like(params)
            
            # Use L-BFGS-B on tridiagonal parameters
            res_tri = minimize(
                tri_objective,
                tri_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.isfinite(res_tri.fun):
                roots_tri = _tridiag_to_roots(res_tri.x)
                if roots_tri is not None:
                    candidate = np.sort(roots_tri)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with perturbed initializations
    if time_ok():
        rng = np.random.default_rng(42)
        for trial in range(6):
            if not time_ok():
                break
            # Perturb the best roots
            pert = best_roots * (1.0 + 0.15 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-3)
            pert = np.sort(pert)
            try:
                res = minimize(
                    objective_grad,
                    pert,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-8, None)] * 12,
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if time_ok():
        try:
            res_final = minimize(
                objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-10, None)] * 12,
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.all(res_final.x > 0) and np.isfinite(res_final.fun):
                candidate = np.sort(res_final.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
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
