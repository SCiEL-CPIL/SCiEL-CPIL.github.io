"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 200.0  # Safety margin before 210s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 = A(f) * A(Fourier(f)) for a test function
    f(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that f(0) = 0.
    
    The Hermite polynomials used are physicists' Hermite polynomials.
    The Fourier transform of H_n(x) * exp(-x^2/2) is (-i)^n * H_n(x) * exp(-x^2/2).
    
    A(f) = integral |f(x)| dx
    A(Fourier(f)) = integral |F(f)(xi)| dxi
    
    Since F(f)(xi) = (1/sqrt(2*pi)) * integral f(x) e^{-i x xi} dx
    But for Hermite-Gaussian functions, the Fourier transform has a known form.
    
    Actually, let's be more careful. The standard result is:
    If f(x) = H_n(x) * exp(-x^2/2), then 
    F(f)(xi) = (1/sqrt(2*pi)) * (-i)^n * H_n(xi) * exp(-xi^2/2) * sqrt(2*pi) ... 
    
    Let me use the property that for the normalized Hermite functions:
    h_n(x) = (1/sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2)
    The Fourier transform of h_n is (-i)^n h_n.
    
    So if f(x) = sum c_n H_n(x) exp(-x^2/2), we can write this in terms of h_n.
    
    H_n(x) exp(-x^2/2) = sqrt(2^n n! sqrt(pi)) * h_n(x)
    
    So f(x) = sum c_n sqrt(2^n n! sqrt(pi)) h_n(x)
    
    F(f)(xi) = sum c_n sqrt(2^n n! sqrt(pi)) (-i)^n h_n(xi)
    
    A(f) = integral |f(x)| dx
    A(F(f)) = integral |F(f)(xi)| dxi
    
    We need to numerically compute these integrals.
    
    The constraint f(0) = 0 gives:
    sum c_n H_n(0) = 0
    
    H_0(0) = 1
    H_4(0) = -12
    H_8(0) = 1680
    H_12(0) = -133056
    
    So: c0 * 1 + c4 * (-12) + c8 * 1680 + c12 * (-133056) = 0
    c12 = (c0 - 12*c4 + 1680*c8) / 133056
    """
    # Compute c12 from the constraint f(0) = 0
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -133056.0
    
    c12 = (c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / (-H12_0)
    
    # Define the function
    def f(x):
        return (c0 * eval_hermite(0, x) + 
                c4 * eval_hermite(4, x) + 
                c8 * eval_hermite(8, x) + 
                c12 * eval_hermite(12, x)) * np.exp(-x**2 / 2)
    
    def F_f(xi):
        # Fourier transform: F(f)(xi) = (1/sqrt(2*pi)) * integral f(x) e^{-i x xi} dx
        # Using the property: F[H_n(x) exp(-x^2/2)] = (1/sqrt(2*pi)) * (-i)^n * H_n(xi) * exp(-xi^2/2) * sqrt(2*pi)
        # Actually, let me verify: 
        # integral H_n(x) exp(-x^2/2) e^{-i x xi} dx = (-i)^n sqrt(2*pi) H_n(xi) exp(-xi^2/2)
        # So F(f)(xi) = (1/sqrt(2*pi)) * [c0*(-i)^0*sqrt(2*pi)*H_0(xi)*exp(-xi^2/2) + 
        #                                  c4*(-i)^4*sqrt(2*pi)*H_4(xi)*exp(-xi^2/2) + ...]
        # = c0*H_0(xi)*exp(-xi^2/2) + c4*(-i)^4*H_4(xi)*exp(-xi^2/2) + ...
        # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
        # So all coefficients are real and positive!
        return (c0 * eval_hermite(0, xi) + 
                c4 * eval_hermite(4, xi) + 
                c8 * eval_hermite(8, xi) + 
                c12 * eval_hermite(12, xi)) * np.exp(-xi**2 / 2)
    
    # Numerically integrate |f(x)| and |F_f(xi)|
    # Use a reasonable range, say [-15, 15]
    x = np.linspace(-15, 15, 10000)
    fx = f(x)
    Fxi = F_f(x)
    
    # Trapzoidal rule
    A_f = np.trapz(np.abs(fx), x)
    A_Ff = np.trapz(np.abs(Fxi), x)
    
    return A_f * A_Ff

def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound on C4.
    The degree-12 coefficient is determined by the constraint f(0) = 0.
    """
    # Starting point: the prior state of the art
    x0 = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    
    # Try multiple starting points
    starting_points = [
        x0,
        np.array([-1.0, 0.01, 0.0001]),
        np.array([-0.5, 0.05, 0.001]),
        np.array([-1.5, 0.005, 0.00005]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.0005]),
        np.array([-1.13, 0.03, 0.0003]),
        np.array([-1.13, 0.05, 0.0003]),
    ]
    
    best_bound = np.inf
    best_coeffs = x0.copy()
    
    for sp in starting_points:
        if _time_remaining() < 10:
            break
        
        try:
            result = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                sp,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead as well from the best point found
    if _time_remaining() > 10:
        try:
            result_nm = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result_nm.fun < best_bound:
                best_bound = result_nm.fun
                best_coeffs = result_nm.x.copy()
        except Exception:
            pass
    
    # Final polish with high precision
    if _time_remaining() > 5:
        try:
            result_polish = minimize(
                lambda c: _compute_bound(c[0], c[1], c[2]),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result_polish.fun < best_bound:
                best_bound = result_polish.fun
                best_coeffs = result_polish.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
