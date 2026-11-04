"""Rewritten program using Nelder-Mead simplex optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.fft import fft
import math


def _hermite_polynomial(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2 * x
    
    H_prev = np.ones_like(x, dtype=float)  # H_0
    H_curr = 2 * x  # H_1
    
    for k in range(1, n):
        H_next = 2 * x * H_curr - 2 * k * H_prev
        H_prev = H_curr
        H_curr = H_next
    
    return H_curr


def _compute_coefficients(c0, c4, c8):
    """
    Given three free coefficients for H0, H4, H8, compute the H12 coefficient
    such that g(0) = 0.
    """
    c12 = (c0 - 12.0 * c4 + 1008.0 * c8) / 103950.0
    return c0, c4, c8, c12


def _upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = e^{-x^2/2} [c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)]
    where c12 is chosen so g(0) = 0.
    
    Note: The problem statement implies the test function is a combination of Hermite polynomials
    times a Gaussian. The standard Hermite functions are orthonormal with the Gaussian e^{-x^2/2}.
    We assume g(x) = e^{-x^2/2} P(x).
    """
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    # Analytical L2 norm
    # ||g||_2^2 = integral |e^{-x^2/2} P(x)|^2 dx = integral e^{-x^2} P(x)^2 dx
    # Using orthogonality of Hermite polynomials: integral e^{-x^2} H_n(x) H_m(x) dx = sqrt(pi) 2^n n! delta_nm
    # Wait, the standard orthogonality is for weight e^{-x^2}.
    # integral_{-inf}^{inf} e^{-x^2} H_n(x) H_m(x) dx = sqrt(pi) 2^n n! delta_nm
    # So ||g||_2^2 = sqrt(pi) * (c0^2 * 1 + c4^2 * 2^4 * 4! + c8^2 * 2^8 * 8! + c12^2 * 2^12 * 12!)
    
    norm_sq = 0.0
    # H0: 2^0 * 0! = 1
    norm_sq += c0**2 * 1.0
    # H4: 2^4 * 24 = 16 * 24 = 384
    norm_sq += c4**2 * 384.0
    # H8: 2^8 * 40320 = 256 * 40320 = 10321920
    norm_sq += c8**2 * 10321920.0
    # H12: 2^12 * 12! = 4096 * 479001600 = 1961988556800
    norm_sq += c12**2 * 1961988556800.0
    
    L2_sq = math.sqrt(math.pi) * norm_sq
    
    if L2_sq < 1e-15:
        return 1e10
        
    L2 = math.sqrt(L2_sq)
    
    # Numerical L1 norm using quad
    def g(x):
        H0 = 1.0
        H4 = 16*x**4 - 48*x**2 + 12
        H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
        H12 = 4096*x**12 - 98304*x**10 + 903168*x**8 - 4081920*x**6 + 9676800*x**4 - 9676800*x**2 + 443520
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return math.exp(-0.5 * x * x) * P

    L1, _ = quad(lambda x: abs(g(x)), -np.inf, np.inf, limit=200)
    
    if L1 < 1e-15:
        return 1e10
        
    A_g = L1 / L2
    
    # Compute Fourier transform L1 norm
    # We use FFT on a grid. Since g(x) decays fast, a finite grid is sufficient.
    # We need to be careful with the scaling.
    N = 32768
    x_max = 20.0
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # Evaluate g on the grid
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 98304*x**10 + 903168*x**8 - 4081920*x**6 + 9676800*x**4 - 9676800*x**2 + 443520
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g_vals = np.exp(-0.5 * x**2) * P
    
    G = fft(g_vals) * dx
    dxi = 1.0 / (N * dx)
    
    L1_G = np.sum(np.abs(G)) * dxi
    # L2_G should be equal to L2 by Parseval, but we compute it to be safe/consistent
    # Actually, for the bound A_g * A_G, we use L1_G / L2_G.
    # Since L2_G = L2, we can just use L2.
    A_G = L1_G / L2
    
    bound = A_g * A_G
    
    return bound


def _objective(params):
    """Objective function for optimization: minimize the upper bound."""
    c0, c4, c8 = params
    try:
        val = _upper_bound_C4(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound
    on A(f) * A(Fourier(f)) using Nelder-Mead simplex optimization.
    """
    # Try multiple initial guesses to avoid local minima
    initial_guesses = [
        np.array([-1.13, 0.04, 0.000308]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0004]),
        np.array([-0.5, 0.06, 0.0002]),
        np.array([-2.0, 0.02, 0.0005]),
        np.array([-1.13, 0.04, 0.000308]),
    ]
    
    best_result = None
    best_fun = 1e10
    
    for x0 in initial_guesses:
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'adaptive': True
                }
            )
            
            if np.isfinite(result.fun) and result.fun < best_fun:
                best_fun = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is not None:
        return np.array(best_result)
    else:
        # Fallback to original initial guess
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
