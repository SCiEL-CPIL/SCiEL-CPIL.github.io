"""Eigenvalue-based parameterization for 12 double roots, optimized to minimize C4 bound.
The roots are eigenvalues of a 12x12 symmetric tridiagonal matrix. Optimizing the matrix
elements (diagonal and off-diagonal) gives a smooth manifold where roots are automatically
distinct, ordered, and positive (with appropriate constraints)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre, roots_genlaguerre

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_p_squared_coeffs(roots):
    """Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    from numpy.polynomial.polynomial import polyfromroots, polymul
    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)
    return P, P2

def _compute_c4(roots):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    P, P2 = _build_p_squared_coeffs(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf
    c4 = I2 / (I1 * I1)
    return float(c4)

def _eig_to_params(eigenvalues, off_diagonals):
    """Reconstruct tridiagonal matrix from eigenvalues and off-diagonals, return eigenvalues."""
    n = len(eigenvalues)
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = eigenvalues[i]
    for i in range(n - 1):
        val = off_diagonals[i]
        T[i, i+1] = val
        T[i+1, i] = val
    return T

def _params_to_roots(params):
    """Convert parameter vector to 12 positive roots via tridiagonal eigenvalue decomposition.
    params = [d0, d1, ..., d11, o0, o1, ..., o10] (12 diagonals + 11 off-diagonals)
    """
    n = 12
    d = params[:n]
    o = params[n:]
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = d[i]
    for i in range(n - 1):
        T[i, i+1] = o[i]
        T[i+1, i] = o[i]
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(T)
    return np.sort(eigvals)

def _objective_eig(params):
    """Objective: C4 bound computed from eigenvalues of tridiagonal matrix."""
    try:
        roots = _params_to_roots(params)
        if np.any(roots <= 0) or np.any(np.isnan(roots)):
            return np.inf
        return _compute_c4(roots)
    except Exception:
        return np.inf

def _log_eig_objective(log_params):
    """Objective in log-space for off-diagonals to ensure positive off-diagonals.
    log_params = [d0, d1, ..., d11, log(o0), log(o1), ..., log(o10)]
    """
    try:
        n = 12
        d = log_params[:n]
        log_o = log_params[n:]
        o = np.exp(log_o)
        params = np.concatenate([d, o])
        return _objective_eig(params)
    except Exception:
        return np.inf

def construct():
    start = time.time()
    DEADLINE = 620.0
    SAFETY = 10.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf

    # Phase 1: Get initial roots from generalized Laguerre nodes
    alpha_values = np.concatenate([
        np.logspace(-2, 3, 50),
        np.linspace(0.1, 50, 100),
    ])
    alpha_values = np.unique(np.sort(alpha_values))
    
    best_alpha = 1.0
    for alpha in alpha_values:
        if not time_ok():
            break
        try:
            xg, _ = roots_genlaguerre(12, float(alpha))
            xg = np.sort(xg)
            if np.all(xg > 0) and np.isfinite(xg).all():
                c4g = _compute_c4(xg)
                if np.isfinite(c4g) and c4g < best_c4:
                    best_c4 = c4g
                    best_roots = xg.copy()
                    best_alpha = float(alpha)
        except Exception:
            pass

    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)

    # Phase 2: Direct L-BFGS-B optimization in log-space on roots
    def _log_roots_objective(log_roots):
        try:
            roots = np.exp(log_roots)
            return _compute_c4(roots)
        except Exception:
            return np.inf

    def _log_roots_grad(log_roots):
        """Compute objective and gradient in log-space."""
        try:
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(np.isnan(roots)):
                return np.inf, np.zeros_like(log_roots)
            
            from numpy.polynomial.polynomial import polyfromroots, polymul, polydiv
            Q = polyfromroots(roots)
            P = polymul(Q, Q)
            P2 = polymul(P, P)
            
            I1 = _poly_moment(P)
            I2 = _poly_moment(P2)
            if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
                return np.inf, np.zeros_like(log_roots)
            
            c4 = I2 / (I1 * I1)
            
            # Gradient: dC4/dx_i where P = Q^2, Q = prod(x - x_j)
            # dQ/dx_i = prod_{j!=i}(x-x_j) = Q/(x-x_i)
            # dP/dx_i = 2*Q*dQ/dx_i = 2*Q^2/(x-x_i) = 2*P/(x-x_i)
            # I1 = int P e^{-x} dx, dI1/dx_i = 2*int [P/(x-x_i)] e^{-x} dx
            # I2 = int P^2 e^{-x} dx, dI2/dx_i = 4*int [P^2/(x-x_i)] e^{-x} dx
            # dC4/dx_i = dI2/I1^2 - 2*I2*dI1/I1^3
            # Chain rule: dC4/d(log x_i) = dC4/dx_i * x_i
            
            n = len(roots)
            dI1 = np.zeros(n)
            dI2 = np.zeros(n)
            
            for i in range(n):
                divisor = np.array([-roots[i], 1.0])
                S_i, _ = polydiv(P, divisor)
                T_i, _ = polydiv(P2, divisor)
                dI1[i] = 2.0 * _poly_moment(S_i)
                dI2[i] = 4.0 * _poly_moment(T_i)
            
            I1_sq = I1 * I1
            I1_cu = I1_sq * I1
            grad_x = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
            grad_log = grad_x * roots
            
            if not np.all(np.isfinite(grad_log)):
                return np.inf, np.zeros_like(log_roots)
            
            return float(c4), grad_log
        except Exception:
            return np.inf, np.zeros_like(log_roots)

    if time_ok():
        try:
            log_init = np.log(best_roots.copy())
            res = minimize(
                _log_roots_objective,
                log_init,
                jac=_log_roots_grad,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Eigenvalue-based optimization
    # Initialize tridiagonal matrix from current best roots
    if time_ok():
        try:
            # Try to construct a tridiagonal matrix whose eigenvalues are best_roots
            # Use a simple initialization: set diagonals to mean of adjacent roots,
            # off-diagonals to geometric mean of adjacent spacings
            n = 12
            d_init = np.zeros(n)
            o_init = np.zeros(n - 1)
            
            for i in range(n):
                if i == 0:
                    d_init[i] = best_roots[0]
                elif i == n - 1:
                    d_init[i] = best_roots[n - 1]
                else:
                    d_init[i] = 0.5 * (best_roots[i] + best_roots[i+1])
            
            for i in range(n - 1):
                spacing = best_roots[i+1] - best_roots[i]
                if spacing > 0:
                    o_init[i] = math.sqrt(spacing * spacing)  # just spacing
                else:
                    o_init[i] = 0.1
            
            # Ensure positive off-diagonals
            o_init = np.maximum(o_init, 1e-3)
            
            # Use log-space for off-diagonals
            log_o_init = np.log(o_init)
            params_init = np.concatenate([d_init, log_o_init])
            
            def _eig_obj_grad(params):
                """Objective and gradient for eigenvalue parameterization."""
                try:
                    n = 12
                    d = params[:n]
                    log_o = params[n:]
                    o = np.exp(log_o)
                    
                    T = np.zeros((n, n))
                    for i in range(n):
                        T[i, i] = d[i]
                    for i in range(n - 1):
                        T[i, i+1] = o[i]
                        T[i+1, i] = o[i]
                    
                    eigvals, eigvecs = np.linalg.eigh(T)
                    eigvals = np.sort(eigvals)
                    
                    if np.any(eigvals <= 0) or np.any(np.isnan(eigvals)):
                        return np.inf, np.zeros_like(params)
                    
                    c4 = _compute_c4(eigvals)
                    if not np.isfinite(c4):
                        return np.inf, np.zeros_like(params)
                    
                    # Gradient via automatic differentiation-like approach
                    # d(eigenvalue_i)/d(T_jk) = eigvecs[j,i] * eigvecs[k,i]
                    # We need dC4/d(eigvals) first, then chain rule
                    
                    from numpy.polynomial.polynomial import polyfromroots, polymul, polydiv
                    Q = polyfromroots(eigvals)
                    P = polymul(Q, Q)
                    P2 = polymul(P, P)
                    
                    I1 = _poly_moment(P)
                    I2 = _poly_moment(P2)
                    if I1 <= 0:
                        return np.inf, np.zeros_like(params)
                    
                    # dC4/d(eigval_i)
                    dI1 = np.zeros(n)
                    dI2 = np.zeros(n)
                    for i in range(n):
                        divisor = np.array([-eigvals[i], 1.0])
                        S_i, _ = polydiv(P, divisor)
                        T_i, _ = polydiv(P2, divisor)
                        dI1[i] = 2.0 * _poly_moment(S_i)
                        dI2[i] = 4.0 * _poly_moment(T_i)
                    
                    I1_sq = I1 * I1
                    I1_cu = I1_sq * I1
                    grad_eig = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
                    
                    # Now chain rule: dC4/d(T_jk) = sum_i grad_eig[i] * d(eig_i)/d(T_jk)
                    # d(eig_i)/d(T_jk) = V[j,i] * V[k,i]
                    grad_T = np.zeros((n, n))
                    for i in range(n):
                        grad_T += grad_eig[i] * np.outer(eigvecs[:, i], eigvecs[:, i])
                    
                    # Now map back to params
                    grad_params = np.zeros_like(params)
                    # Diagonal: dC4/d(d_i) = grad_T[i,i]
                    for i in range(n):
                        grad_params[i] = grad_T[i, i]
                    # Off-diagonal: dC4/d(o_i) = grad_T[i, i+1] + grad_T[i+1, i] = 2*grad_T[i, i+1]
                    # Chain rule for log_o: dC4/d(log_o_i) = dC4/d(o_i) * o_i
                    for i in range(n - 1):
                        grad_params[n + i] = 2.0 * grad_T[i, i+1] * o[i]
                    
                    if not np.all(np.isfinite(grad_params)):
                        return np.inf, np.zeros_like(params)
                    
                    return float(c4), grad_params
                except Exception:
                    return np.inf, np.zeros_like(params)
            
            res = minimize(
                _eig_obj_grad,
                params_init,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                d_final = res.x[:12]
                log_o_final = res.x[12:]
                o_final = np.exp(log_o_final)
                T_final = np.zeros((12, 12))
                for i in range(12):
                    T_final[i, i] = d_final[i]
                for i in range(11):
                    T_final[i, i+1] = o_final[i]
                    T_final[i+1, i] = o_final[i]
                eigvals_final = np.sort(np.linalg.eigvalsh(T_final))
                if np.all(eigvals_final > 0):
                    c4 = _compute_c4(eigvals_final)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = eigvals_final.copy()
        except Exception:
            pass

    # Phase 4: Multi-start L-BFGS-B in log-space from perturbed best
    if time_ok():
        log_best = np.log(best_roots.copy())
        candidates_to_refine = [log_best.copy()]
        rng = np.random.default_rng(42)
        
        for scale in [0.02, 0.05, 0.1, 0.2, 0.5]:
            for _ in range(3):
                pert = log_best + scale * rng.standard_normal(12)
                candidates_to_refine.append(pert.copy())
        
        for factor in [0.5, 1.5, 2.0, 0.3, 3.0]:
            pert = log_best + np.log(factor)
            candidates_to_refine.append(pert.copy())
        
        for _ in range(5):
            pert = rng.uniform(-5, 5, size=12)
            candidates_to_refine.append(pert)
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _log_roots_objective,
                    init,
                    jac=_log_roots_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 50}
                )
                if np.isfinite(res.fun):
                    candidate = np.sort(np.exp(res.x))
                    if np.all(candidate > 0):
                        c4 = _compute_c4(candidate)
                        if c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    for polish_iter in range(15):
        if not time_ok():
            break
        try:
            log_best = np.log(best_roots)
            res = minimize(
                _log_roots_objective,
                log_best,
                jac=_log_roots_grad,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100}
            )
            if np.isfinite(res.fun):
                candidate = np.sort(np.exp(res.x))
                if np.all(candidate > 0):
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
