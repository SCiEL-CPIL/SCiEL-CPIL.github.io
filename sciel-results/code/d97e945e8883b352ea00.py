# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import roots_genlaguerre
import time
import math

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses generalized Laguerre quadrature nodes with tunable alpha parameter.
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
        
    M = A_T_inv
    w = np.arange(1, 14)

    def compute_norm_sq_from_roots(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        c = M @ p
        norm_sq = np.sum(c**2 * w)
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def get_roots_for_alpha(alpha):
        """
        Get 12 positive roots of the generalized Laguerre polynomial L_12^(alpha)(x).
        """
        try:
            nodes, _ = roots_genlaguerre(12, alpha)
            # Ensure positive (numerical safety)
            nodes = np.maximum(nodes, 1e-10)
            return nodes
        except Exception:
            return None

    def cost_from_alpha(alpha):
        """
        Compute C4 upper bound (1/norm_sq) for given alpha.
        """
        roots = get_roots_for_alpha(alpha)
        if roots is None:
            return 1e10
        norm_sq = compute_norm_sq_from_roots(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Strategy 1: Optimize alpha in 1D using scalar minimization
    best_val = 1e10
    best_alpha = None
    best_roots = None

    # Scan a range of alpha values to find good regions
    alpha_values = np.linspace(-0.5, 50.0, 1000)
    vals = []
    for a in alpha_values:
        if not time_ok():
            break
        v = cost_from_alpha(a)
        vals.append(v)
    
    if vals:
        vals = np.array(vals)
        min_idx = np.argmin(vals)
        best_alpha = alpha_values[min_idx]
        best_val = vals[min_idx]
        best_roots = get_roots_for_alpha(best_alpha)
        
        # Refine around the best alpha
        if best_alpha is not None and time_ok():
            # Local refinement
            lo = max(-0.5, best_alpha - 2.0)
            hi = min(50.0, best_alpha + 2.0)
            try:
                res = minimize_scalar(
                    cost_from_alpha,
                    bounds=(lo, hi),
                    method='bounded',
                    options={'maxiter': 200, 'xatol': 1e-10}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_alpha = res.x
                    best_roots = get_roots_for_alpha(best_alpha)
            except Exception:
                pass

    # Strategy 2: If time remains, do a 12-DOF polish starting from the quadrature nodes
    if best_roots is not None and time_ok():
        def cost_and_grad_12d(log_roots):
            """
            Compute cost (1/norm_sq) and gradient w.r.t log_roots for 12-DOF optimization.
            """
            roots = np.exp(log_roots)
            
            if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
                return 1e10, np.zeros(12)
                
            norm_sq = compute_norm_sq_from_roots(roots)
            
            if norm_sq <= 1e-10:
                return 1e10, np.zeros(12)
                
            cost = 1.0 / norm_sq
            
            # Gradient calculation
            coeffs_desc = np.poly(roots)
            p = coeffs_desc[::-1]
            c = M @ p
            
            grad_r = np.zeros(12)
            for i in range(12):
                roots_minus_i = np.delete(roots, i)
                q_coeffs_desc = np.poly(roots_minus_i)
                q = q_coeffs_desc[::-1]
                dp_dr_i = -q
                v = M @ dp_dr_i
                d_norm_sq = 2.0 * np.sum(c * w * v)
                grad_r[i] = -1.0 / (norm_sq**2) * d_norm_sq
            
            grad_y = grad_r * roots
            return cost, grad_y

        # Start from the quadrature nodes
        init_log_roots = np.log(best_roots)
        bounds_log = [(-4.0, 10.0)] * 12
        
        try:
            res = minimize(
                cost_and_grad_12d,
                init_log_roots,
                method='L-BFGS-B',
                bounds=bounds_log,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 600}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
        except Exception:
            pass

    # Fallback if nothing worked
    if best_roots is None:
        # Use standard Laguerre nodes (alpha=1)
        nodes, _ = roots_genlaguerre(12, 1.0)
        best_roots = np.maximum(nodes, 1e-10)

    best_roots = np.maximum(best_roots, 1e-6)
    return list(best_roots)
# EVOLVE-BLOCK-END
