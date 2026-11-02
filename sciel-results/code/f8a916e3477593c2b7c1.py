"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite


def _compute_c12(c0, c4, c8):
    """Compute c12 such that g(0) = 0 where g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12."""
    # H_n(0) for physicists' Hermite:
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c12


def _objective(coeffs):
    """Compute the upper bound on C4 for given coefficients [c0, c4, c8]."""
    c0, c4, c8 = coeffs
    c12 = _compute_c12(c0, c4, c8)
    
    x = np.linspace(-15, 15, 10001)
    
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = P * np.exp(-x**2 / 2)
    
    abs_f_sq = np.abs(f)**2
    norm_f_sq = np.trapezoid(abs_f_sq, x)
    
    if norm_f_sq < 1e-30:
        return 1e10
    
    # Compute ||x*f||^2 / ||f||^2
    x2_weighted = x**2 * abs_f_sq
    norm_xf_sq = np.trapezoid(x2_weighted, x)
    ratio_x = norm_xf_sq / norm_f_sq
    
    # Compute ||f'||^2 / ||f||^2 for the k-space variance
    H3 = eval_hermite(3, x)
    H7 = eval_hermite(7, x)
    H11 = eval_hermite(11, x)
    
    P_prime = c4 * 8 * H3 + c8 * 16 * H7 + c12 * 24 * H11
    fp = (P_prime - x * P) * np.exp(-x**2 / 2)
    
    norm_fp_sq = np.trapezoid(np.abs(fp)**2, x)
    ratio_k = norm_fp_sq / norm_f_sq
    
    # The uncertainty product bound
    return ratio_x * ratio_k


def construct():
    """Return optimized coefficients [c0, c4, c8]."""
    start_time = time.time()
    deadline = start_time + 175
    
    best_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _objective(best_coeffs)
    
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        
        scale = max(0.01, 1.0 / (trial + 1))
        perturb = np.random.normal(0, scale, 3)
        x0 = best_coeffs + perturb
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue
    
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
