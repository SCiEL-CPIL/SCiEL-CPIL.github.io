"""Improved program using robust Nelder-Mead simplex optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, fftshift, fftfreq
import math

# Global variables for the grid and precomputed Hermite polynomials
# to speed up objective function evaluations
_GRID = None
_HERMITE_POLYS = None

def _setup_grid(N=32768, L=20.0):
    """
    Set up the computational grid and precompute Hermite polynomials.
    N: Number of points.
    L: Half-length of the interval [-L, L].
    """
    global _GRID, _HERMITE_POLYS
    if _GRID is not None and _GRID[0] == N and _GRID[1] == L:
        return _GRID, _HERMITE_POLYS
    
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    # Precompute H0, H4, H8, H12
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    H0 = np.ones_like(x)
    H1 = 2 * x
    H2 = 4 * x**2 - 2
    H3 = 8 * x**3 - 12 * x
    H4 = 16 * x**4 - 48 * x**2 + 12
    
    H5 = 2 * x * H4 - 8 * H3
    H6 = 2 * x * H5 - 12 * H4
    H7 = 2 * x * H6 - 16 * H5
    H8 = 2 * x * H7 - 20 * H6
    
    H9 = 2 * x * H8 - 24 * H7
    H10 = 2 * x * H9 - 28 * H8
    H11 = 2 * x * H10 - 32 * H9
    H12 = 2 * x * H11 - 36 * H10
    
    _GRID = (N, L, x, dx)
    _HERMITE_POLYS = (H0, H4, H8, H12)
    return _GRID, _HERMITE_POLYS

def _compute_coefficients(c0, c4, c8):
    """
    Compute c12 such that g(0) = 0.
    g(0) = c0*1 + c4*(-12) + c8*(1008) + c12*(-103950) = 0
    c12 = (c0 - 12*c4 + 1008*c8) / 103950
    """
    c12 = (c0 - 12.0 * c4 + 1008.0 * c8) / 103950.0
    return c0, c4, c8, c12

def _objective(params):
    """
    Objective function: minimize A(g) * A(G).
    A(f) = ||f||_1 / ||f||_2
    """
    N, L, x, dx = _GRID
    H0, H4, H8, H12 = _HERMITE_POLYS
    
    c0, c4, c8 = params
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    # Construct g(x)
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # Check for zero function
    max_abs_g = np.max(np.abs(g))
    if max_abs_g < 1e-15:
        return 1e10
        
    # Compute L1 and L2 norms of g using trapezoidal rule
    L1_g = np.trapz(np.abs(g), x)
    L2_g_sq = np.trapz(g**2, x)
    
    if L2_g_sq < 1e-15:
        return 1e10
        
    L2_g = np.sqrt(L2_g_sq)
    A_g = L1_g / L2_g
    
    # Compute Fourier Transform G(ξ)
    # F(ξ) = ∫ g(x) exp(-2πi x ξ) dx
    # Using FFT: G[k] ≈ dx * FFT(g)[k]
    G = fft(g) * dx
    
    # Compute L1 and L2 norms of G
    # Frequency spacing dξ = 1 / (N * dx)
    dxi = 1.0 / (N * dx)
    
    L1_G = np.sum(np.abs(G)) * dxi
    L2_G_sq = np.sum(np.abs(G)**2) * dxi**2
    
    if L2_G_sq < 1e-15:
        return 1e10
        
    L2_G = np.sqrt(L2_G_sq)
    A_G = L1_G / L2_G
    
    bound = A_g * A_G
    
    if not np.isfinite(bound):
        return 1e10
        
    return bound

def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound
    using Nelder-Mead simplex optimization.
    """
    # Setup grid with high resolution for accuracy
    _setup_grid(N=32768, L=20.0)
    
    # Define initial guesses
    # The optimal values are known to be roughly around:
    # c0 ~ -1.13, c4 ~ 0.04, c8 ~ 0.0003
    # We provide several starting points to ensure global convergence
    initial_guesses = [
        np.array([-1.13, 0.04, 0.000308]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.5, 0.03, 0.0004]),
        np.array([-0.8, 0.06, 0.0002]),
        np.array([-1.2, 0.035, 0.00032]),
        np.array([-1.0, 0.0, 0.0]), # Wider search
        np.array([-2.0, 0.1, 0.001]), # Even wider
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
                    'adaptive': True,
                    'initial_simplex': None # Let scipy choose, or we can define one
                }
            )
            
            if np.isfinite(result.fun) and result.fun < best_fun:
                best_fun = result.fun
                best_result = result.x.copy()
        except Exception:
            continue
    
    if best_result is not None:
        return np.array(best_result)
    else:
        # Fallback to a known good approximation
        return np.array([-1.13, 0.04, 0.000308])
# EVOLVE-BLOCK-END
