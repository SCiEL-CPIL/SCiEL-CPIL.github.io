"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Constants for the Hermite-Gaussian test function
# The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
# The coefficient a12 is fixed to impose g(0) = 0.
# H0(0) = 1, H4(0) = -3, H8(0) = 105, H12(0) = -10395
# So a12 = -(a0 - 3*a4 + 105*a8) / (-10395) = (a0 - 3*a4 + 105*a8) / 10395

def _coeff_a12(a0, a4, a8):
    return (a0 - 3*a4 + 105*a8) / 10395.0

def _compute_bound(params):
    """
    Compute the upper bound C4 for the given Hermite coefficients.
    params: array [a0, a4, a8]
    """
    a0, a4, a8 = params
    a12 = _coeff_a12(a0, a4, a8)
    
    # We need to compute A(f) * A(Fourier(f))
    # A(f) = sup_x |f(x)| / ||f||_2
    # For a Hermite-Gaussian polynomial, we can compute this numerically.
    # The Fourier transform of H_n(x) * exp(-x^2/2) is (-i)^n H_n(x) * exp(-x^2/2)
    # So |Fourier(f)(x)| = |f(x)|, meaning A(f) = A(Fourier(f))
    # Thus A(f)*A(Fourier(f)) = A(f)^2
    
    # A(f) = sup_x |g(x)| / ||g||_2
    # where g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    
    # We need to find sup_x |g(x)| and ||g||_2
    # ||g||_2^2 = integral |g(x)|^2 dx
    
    # For Hermite polynomials, the integral of H_n(x) H_m(x) exp(-x^2) dx = 2^n n! sqrt(pi) delta_nm
    # But here we have exp(-x^2/2) in the function, so |g(x)|^2 = P(x)^2 exp(-x^2)
    # where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    
    # ||g||_2^2 = integral P(x)^2 exp(-x^2) dx
    # This can be computed using the orthogonality of Hermite polynomials.
    # P(x)^2 = sum_{i,j} a_i a_j H_i(x) H_j(x) where i,j in {0,4,8,12}
    # integral H_i(x) H_j(x) exp(-x^2) dx = 2^i i! sqrt(pi) delta_ij
    
    # So ||g||_2^2 = sqrt(pi) * (a0^2 * 2^0 * 0! + a4^2 * 2^4 * 4! + a8^2 * 2^8 * 8! + a12^2 * 2^12 * 12!)
    
    norm_sq = np.sqrt(np.pi) * (a0**2 * 1 * 1 + a4**2 * 16 * 24 + a8**2 * 256 * 40320 + a12**2 * 4096 * 479001600)
    norm = np.sqrt(norm_sq)
    
    # Now we need sup_x |g(x)| = sup_x |P(x) exp(-x^2/2)|
    # This requires numerical optimization over x.
    
    def _g_abs(x):
        px = a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x) + a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)
        return abs(px * np.exp(-x**2 / 2))
    
    # Find the maximum of |g(x)| by minimizing -|g(x)|
    # We'll search over a reasonable range of x
    # The maximum should be near x=0 or at some finite x
    
    best_val = 0.0
    # Use a grid to find approximate location, then refine
    x_grid = np.linspace(-10, 10, 1000)
    vals = [_g_abs(x) for x in x_grid]
    max_idx = np.argmax(vals)
    x_start = x_grid[max_idx]
    
    # Refine with scipy minimize
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda x: -_g_abs(x), bounds=(x_start - 1, x_start + 1), method='bounded', options={'xatol': 1e-10})
    max_abs = -result.fun
    
    A_f = max_abs / norm
    bound = A_f ** 2
    
    return bound

def _gradient(params):
    """
    Compute the gradient of the bound with respect to params [a0, a4, a8].
    """
    eps = 1e-7
    grad = np.zeros(3)
    f0 = _compute_bound(params)
    for i in range(3):
        params_pert = params.copy()
        params_pert[i] += eps
        f_pert = _compute_bound(params_pert)
        grad[i] = (f_pert - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = 180.0  # Must finish within 180s, keep safety margin
    
    # Initial guess from the prior state of the art
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound(best_params)
    
    # Try multiple starting points
    start_points = [
        best_params,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-1.1, 0.05, 0.001]),
        np.array([-0.9, 0.02, 0.0005]),
        np.array([-1.2, 0.03, 0.0003]),
        np.array([-1.05, 0.04, 0.0002]),
    ]
    
    for sp in start_points:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            result = minimize(
                _compute_bound,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _compute_bound,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
