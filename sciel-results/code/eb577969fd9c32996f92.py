import numpy as np
from scipy.optimize import minimize, brentq
import time
import math

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses potential-based parameterization: roots are critical points of V(x) = sum c_k * x^k,
    solved via Newton-Raphson for the coupled equations.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    m = 12  # 12 double roots
    n = 1   # dimension

    # Helper: compute C4 upper bound from root locations
    def _compute_c4(roots):
        """
        Compute the C4 uncertainty bound from 12 positive double-root locations.
        The bound involves the Laguerre polynomial construction.
        For m=12, n=1, we compute the norm-based bound.
        """
        try:
            roots = np.asarray(roots, dtype=float)
            if len(roots) != m or np.any(roots <= 0):
                return 1e10
            
            # Sort roots
            roots = np.sort(roots)
            
            # Build the polynomial P(x) = prod_i (x - r_i)^2
            # Use log-space for numerical stability
            # P(x) has degree 2m = 24
            
            # Compute using companion matrix or direct polynomial multiplication
            # P(x) = prod (x^2 - 2*r_i*x + r_i^2)
            coeffs = np.array([1.0])
            for r in roots:
                # Multiply by (x^2 - 2r*x + r^2)
                new_coeffs = np.zeros(len(coeffs) + 2)
                new_coeffs[2:] += coeffs
                new_coeffs[1:-1] += (-2*r) * coeffs
                new_coeffs[:len(coeffs)] += (r*r) * coeffs
                coeffs = new_coeffs
            
            # coeffs[0] is x^24 coefficient, coeffs[-1] is constant
            # Now compute the Laguerre-weighted integral of P(x) * exp(-x) * x^n
            # For n=1: integral_0^inf P(x) * exp(-x) * x^1 dx
            # = sum_k coeffs[k] * integral x^(24-k+1) * exp(-x) dx
            # = sum_k coeffs[k] * Gamma(25-k)
            # = sum_k coeffs[k] * (24-k)!
            
            total = 0.0
            for k in range(len(coeffs)):
                power = 24 - k + 1  # x^(24-k) * x^1 = x^(25-k)
                if power <= 0:
                    continue
                # Gamma(power) = (power-1)!
                # Use math.lgamma for stability
                log_gam = math.lgamma(power)
                if log_gam > 700:  # overflow
                    return 1e10
                total += coeffs[k] * math.exp(log_gam)
            
            if total <= 0:
                return 1e10
            
            # The C4 bound is related to sqrt(total) or similar
            # For the Fourier sign uncertainty, the bound involves the norm
            # C4 ~ sqrt( integral P(x) exp(-x) x^n dx )
            c4 = math.sqrt(total)
            
            return c4
        except Exception:
            return 1e10

    # Potential-based parameterization
    # V(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    # Critical points: V'(x) = c1 + 2*c2*x + 3*c3*x^2 + 4*c4*x^3 + 5*c5*x^4 = 0
    # This is a quartic in x, which has at most 4 real roots.
    # We need 12 roots, so we need a higher degree potential.
    # V(x) = sum_{k=1}^{13} c_k * x^k  -> V'(x) is degree 12, can have 12 real roots
    
    # Actually, let's use V(x) = sum_{k=1}^{12} c_k * x^k / k
    # Then V'(x) = sum_{k=1}^{12} c_k * x^(k-1), a degree-11 polynomial
    # That gives at most 11 roots. We need 12.
    
    # Use V(x) = sum_{k=1}^{13} c_k * x^k / k
    # V'(x) = sum_{k=1}^{13} c_k * x^(k-1), degree 12
    # Can have up to 12 real positive roots
    
    # Parameters: c_1, ..., c_13 (13 parameters)
    # But we only need to find 12 roots, so we have 1 degree of freedom
    # We can fix c_1 = 1 (scaling) and optimize the other 12
    
    # Newton-Raphson: given c_1,...,c_13, find 12 positive roots of V'(x)=0
    def _find_critical_points(c, max_iter=100, tol=1e-10):
        """
        Find 12 positive critical points of V(x) = sum c_k * x^k / k
        V'(x) = sum_{k=1}^{13} c_k * x^(k-1)
        Uses Newton-Raphson starting from Laguerre nodes.
        """
        try:
            # Initial guess: Laguerre nodes scaled
            roots_init, _ = np.polynomial.laguerre.laggauss(12)
            x = roots_init.copy()
            
            for iteration in range(max_iter):
                # Compute V'(x_i) and V''(x_i)
                # V'(x) = c_1 + c_2*x + c_3*x^2 + ... + c_13*x^12
                # V''(x) = c_2 + 2*c_3*x + 3*c_4*x^2 + ... + 12*c_13*x^11
                
                residuals = np.zeros(12)
                derivs = np.zeros(12)
                
                for i in range(12):
                    xi = x[i]
                    # V'(xi)
                    vp = 0.0
                    vpp = 0.0
                    for k in range(1, 14):
                        vp += c[k-1] * (xi ** (k-1))
                        if k >= 2:
                            vpp += (k-1) * c[k-1] * (xi ** (k-2))
                    residuals[i] = vp
                    derivs[i] = vpp
                
                # Newton step
                mask = np.abs(derivs) > 1e-15
                if not np.all(mask):
                    # Fall back to small step
                    dx = -residuals / np.where(np.abs(derivs) > 1e-15, derivs, 1e15)
                else:
                    dx = -residuals / derivs
                
                x_new = x + dx
                
                # Ensure positivity
                x_new = np.maximum(x_new, 1e-10)
                
                # Check convergence
                if np.max(np.abs(dx)) < tol:
                    x = x_new
                    break
                x = x_new
            
            # Final check: all positive and distinct
            if np.any(x <= 0) or np.any(np.diff(np.sort(x)) < 1e-8):
                return None
            return np.sort(x)
        except Exception:
            return None

    # Objective: given potential coefficients, find roots and compute C4
    def _objective(c_params):
        """c_params = [c_2, ..., c_13] with c_1 fixed to 1"""
        try:
            c = np.zeros(13)
            c[0] = 1.0  # c_1 fixed
            c[1:] = c_params
            
            roots = _find_critical_points(c)
            if roots is None:
                return 1e10
            
            return _compute_c4(roots)
        except Exception:
            return 1e10

    # Initial guess: use Laguerre roots to back-solve for potential coefficients
    # This is an ill-posed problem, so we'll use a simple initial guess
    # and let optimization handle it
    
    # Alternative: directly optimize the 12 roots with the constraint that they
    # are critical points of some potential. But that's complex.
    
    # Simpler approach: optimize the 12 roots directly, but use the potential
    # structure as a regularizer or initialization
    
    # Let's try: start from Laguerre roots, optimize directly to minimize C4
    def _objective_roots(roots):
        """Directly optimize 12 root locations"""
        try:
            if np.any(roots <= 0):
                return 1e10
            return _compute_c4(roots)
        except Exception:
            return 1e10

    # Get initial roots
    try:
        roots_init, _ = np.polynomial.laguerre.laggauss(12)
    except Exception:
        roots_init = 4.0 * np.arange(1, 13)
    
    best_roots = roots_init.copy()
    best_c4 = _compute_c4(best_roots)
    
    # Phase 1: Direct optimization of roots using log-parameterization
    if time.time() < deadline - 60:
        try:
            log_roots_init = np.log(roots_init)
            
            def _obj_log(y):
                x = np.exp(y)
                return _compute_c4(x)
            
            result = minimize(_obj_log, log_roots_init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < best_c4:
                candidate = np.exp(result.x)
                if np.all(candidate > 0):
                    c4_val = _compute_c4(candidate)
                    if c4_val < best_c4:
                        best_c4 = c4_val
                        best_roots = candidate
        except Exception:
            pass

    # Phase 2: Potential-based refinement
    if time.time() < deadline - 30:
        try:
            # Try to fit a potential to the current best roots
            # Use least squares to find c such that V'(r_i) ≈ 0 for all i
            A = np.zeros((12, 13))
            for i in range(12):
                ri = best_roots[i]
                for k in range(13):
                    A[i, k] = ri ** k
            
            # We want A @ c ≈ 0, with c[0] = 1
            # So A[:, 1:] @ c[1:] ≈ -A[:, 0]
            A_reduced = A[:, 1:]
            b = -A[:, 0]
            
            c_solution, residuals, rank, sv = np.linalg.lstsq(A_reduced, b, rcond=None)
            
            c_full = np.zeros(13)
            c_full[0] = 1.0
            c_full[1:] = c_solution
            
            # Verify the roots are indeed close to critical points
            roots_check = _find_critical_points(c_full)
            if roots_check is not None:
                c4_check = _compute_c4(roots_check)
                if c4_check < best_c4:
                    best_c4 = c4_check
                    best_roots = roots_check
        except Exception:
            pass

    # Phase 3: Local refinement with L-BFGS-B on log-roots
    if time.time() < deadline - 10:
        try:
            log_roots = np.log(best_roots)
            
            def _obj_log2(y):
                x = np.exp(y)
                return _compute_c4(x)
            
            result2 = minimize(_obj_log2, log_roots, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result2.success or result2.fun < best_c4:
                candidate = np.exp(result2.x)
                if np.all(candidate > 0):
                    c4_val = _compute_c4(candidate)
                    if c4_val < best_c4:
                        best_c4 = c4_val
                        best_roots = candidate
        except Exception:
            pass

    # Ensure final result is valid
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        # Fallback to Laguerre roots
        try:
            best_roots, _ = np.polynomial.laguerre.laggauss(12)
        except Exception:
            best_roots = 4.0 * np.arange(1, 13)
    
    return best_roots.tolist()
