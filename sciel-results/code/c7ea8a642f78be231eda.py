"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import warnings


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the uncertainty product analytically.
    
    Since Hermite indices {0,4,8,12} differ by >2, x^2 matrix is diagonal.
    M_g = M_F = sum a_n^2 * (n + 0.5)
    Objective = (M_g / N)^2
    """
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 665280.0
    
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # A_n = sqrt(2^n * n! * sqrt(pi))
    A0 = np.sqrt(np.sqrt(np.pi))
    A4 = np.sqrt(16.0 * 24.0 * np.sqrt(np.pi))
    A8 = np.sqrt(256.0 * 40320.0 * np.sqrt(np.pi))
    A12 = np.sqrt(4096.0 * 479001600.0 * np.sqrt(np.pi))
    
    a0 = c0 * A0
    a4 = c4 * A4
    a8 = c8 * A8
    a12 = c12 * A12
    
    N = a0**2 + a4**2 + a8**2 + a12**2
    if N < 1e-30:
        return 1e10
    
    D0, D4, D8, D12 = 0.5, 4.5, 8.5, 12.5
    M_g = D0 * a0**2 + D4 * a4**2 + D8 * a8**2 + D12 * a12**2
    J = (M_g / N) ** 2
    
    return J


def _objective(coeffs):
    """Objective function for optimization - returns the square root of the bound for smoother optimization."""
    c0, c4, c8 = coeffs
    try:
        bound = _compute_bound_analytic(c0, c4, c8)
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        # Return sqrt for smoother landscape (equivalent optimization)
        return np.sqrt(bound)
    except Exception:
        return 1e10


def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    
    Returns an array of three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient is determined to ensure g(0) = 0.
    
    Key insight: The problem reduces to minimizing (weighted avg of D_n)^2 where
    D = [0.5, 4.5, 8.5, 12.5] and weights are a_n^2 / N.
    The constraint g(0)=0 is linear in c's, so we can parametrize 2 free variables.
    """
    # Precompute constants
    A0 = np.sqrt(np.sqrt(np.pi))
    A4 = np.sqrt(16.0 * 24.0 * np.sqrt(np.pi))
    A8 = np.sqrt(256.0 * 40320.0 * np.sqrt(np.pi))
    A12 = np.sqrt(4096.0 * 479001600.0 * np.sqrt(np.pi))
    
    h0_0, h4_0, h8_0, h12_0 = 1.0, 12.0, 1680.0, 665280.0
    D0, D4, D8, D12 = 0.5, 4.5, 8.5, 12.5
    
    def bound_from_params(c0, c4, c8):
        c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
        a0 = c0 * A0
        a4 = c4 * A4
        a8 = c8 * A8
        a12 = c12 * A12
        N = a0**2 + a4**2 + a8**2 + a12**2
        if N < 1e-30:
            return 1e10
        Mg = D0*a0**2 + D4*a4**2 + D8*a8**2 + D12*a12**2
        return (Mg / N)**2
    
    # Parametrize: let c4 = s * c0, c8 = t * c0 (scale invariant)
    # Then c12 = -c0*(1 + 12s + 1680t)/665280
    # The objective is scale-invariant, so we can set c0 = 1 and optimize s, t
    # But this misses solutions where c0 = 0. Let's use a different parametrization.
    
    # Better: use the fact that the problem is homogeneous. 
    # Let u = c4/c0, v = c8/c0 when c0 != 0.
    # Also consider cases where some coefficients are zero.
    
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_bound = bound_from_params(*best_coeffs)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Strategy 1: Optimize in (u, v) = (c4/c0, c8/c0) space
        def obj_uv(params):
            u, v = params
            c0, c4, c8 = 1.0, u, v
            b = bound_from_params(c0, c4, c8)
            if not np.isfinite(b):
                return 1e10
            return np.sqrt(b)  # smoother
        
        np.random.seed(42)
        for trial in range(50):
            if trial == 0:
                # From current best
                c0, c4, c8 = best_coeffs
                if abs(c0) > 1e-10:
                    x0 = np.array([c4/c0, c8/c0])
                else:
                    x0 = np.random.uniform([-2, -2], [2, 2])
            else:
                x0 = np.random.uniform([-3, -3], [3, 3])
            
            res = minimize(obj_uv, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if res.fun < np.sqrt(best_bound):
                u, v = res.x
                c0, c4, c8 = 1.0, u, v
                b = bound_from_params(c0, c4, c8)
                if b < best_bound and np.isfinite(b):
                    best_bound = b
                    best_coeffs = np.array([c0, c4, c8])
        
        # Strategy 2: Direct optimization in (c0, c4, c8) with multiple starts
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        for trial in range(30):
            if trial == 0:
                x0 = best_coeffs.copy()
            elif trial == 1:
                x0 = np.array([-1.0, 0.1, 0.01])
            elif trial == 2:
                x0 = np.array([1.0, -0.1, -0.01])
            elif trial == 3:
                x0 = np.array([0.0, 1.0, 0.1])
            elif trial == 4:
                x0 = np.array([0.0, 0.0, 1.0])
            else:
                x0 = np.random.uniform([-3, -3, -3], [3, 3, 3])
            
            res = minimize(_objective, x0, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-12})
            
            b = bound_from_params(*res.x)
            if b < best_bound and np.isfinite(b):
                best_bound = b
                best_coeffs = res.x.copy()
            
            # Also try Nelder-Mead refinement
            res2 = minimize(_objective, res.x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            b2 = bound_from_params(*res2.x)
            if b2 < best_bound and np.isfinite(b2):
                best_bound = b2
                best_coeffs = res2.x.copy()
        
        # Strategy 3: Exploit structure - try to make a12 and a8 small
        # The constraint is: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
        # We want to minimize weighted avg of D, so maximize weight on a0.
        # This means we want |a0| >> |a4|, |a8|, |a12|
        # Since a_n = c_n * A_n, and A_n grows fast, we need c_n to decrease fast.
        
        # Try grid-like search in log space for ratio parameters
        for log_c4_ratio in np.linspace(-3, 1, 30):
            for log_c8_ratio in np.linspace(-3, 0, 20):
                c4_ratio = np.exp(log_c4_ratio)
                c8_ratio = np.exp(log_c8_ratio)
                for c0_sign in [-1, 1]:
                    c0 = c0_sign
                    c4 = c0_sign * c4_ratio
                    c8 = c0_sign * c8_ratio
                    b = bound_from_params(c0, c4, c8)
                    if b < best_bound and np.isfinite(b):
                        best_bound = b
                        best_coeffs = np.array([c0, c4, c8])
        
        # Final refinement around best
        for _ in range(20):
            x0 = best_coeffs + np.random.normal(0, 0.01, 3)
            res = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
            b = bound_from_params(*res.x)
            if b < best_bound and np.isfinite(b):
                best_bound = b
                best_coeffs = res.x.copy()
    
    final_bound = bound_from_params(*best_coeffs)
    if not np.isfinite(final_bound) or final_bound <= 0:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
