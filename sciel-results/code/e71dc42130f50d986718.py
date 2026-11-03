"""Rewritten program using Nelder-Mead simplex optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import math


def _hermite_polynomial(n, x):
    """Compute the physicists' Hermite polynomial H_n(x)."""
    # Use recurrence relation: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
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
    
    g(x) = c0 * H0(x) + c4 * H4(x) + c8 * H8(x) + c12 * H12(x)
    
    At x=0:
    H0(0) = 1
    H4(0) = -12
    H8(0) = 1008
    H12(0) = -103950
    
    So: c0 * 1 + c4 * (-12) + c8 * 1008 + c12 * (-103950) = 0
    => c12 = (c0 - 12*c4 + 1008*c8) / 103950
    """
    c12 = (c0 - 12.0 * c4 + 1008.0 * c8) / 103950.0
    return c0, c4, c8, c12


def _upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x) where c12 is chosen
    so g(0) = 0.
    
    The uncertainty principle bound involves the L1 norm of g and its Fourier transform.
    For a function g, A(g) = ||g||_1 / ||g||_2 (or similar normalized quantity).
    
    Actually, in the uncertainty inequality context, we need to compute:
    A(f) = ||f||_1 / ||f||_2  (or the appropriate norm ratio)
    
    The product A(f) * A(F(f)) gives the uncertainty bound.
    
    For the Hermite-Gaussian test functions, we can compute these numerically.
    """
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    # Evaluate g(x) on a fine grid
    x = np.linspace(-20, 20, 10000)
    H0 = _hermite_polynomial(0, x)
    H4 = _hermite_polynomial(4, x)
    H8 = _hermite_polynomial(8, x)
    H12 = _hermite_polynomial(12, x)
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    
    # Check if g is essentially zero
    if np.max(np.abs(g)) < 1e-15:
        return 1e10  # Invalid
    
    # Compute L1 and L2 norms
    dx = x[1] - x[0]
    L1 = np.sum(np.abs(g)) * dx
    L2 = np.sqrt(np.sum(g**2)) * dx
    
    if L2 < 1e-15:
        return 1e10
    
    A_g = L1 / L2
    
    # Compute Fourier transform of g
    # The Fourier transform of a function can be computed via FFT
    # But we need to be careful about the definition.
    # Using the standard Fourier transform: F(f)(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    # Or the angular frequency version: integral f(x) exp(-i*x*xi) dx
    
    # Let's use scipy's fft or direct integration
    # For better accuracy, let's use a wider grid and proper FFT
    
    N = len(x)
    # Use FFT to compute the Fourier transform
    # The FFT computes the DFT, which approximates the FT with appropriate scaling
    
    # Actually, let's use a more direct approach with scipy
    from scipy.fft import fft, fftfreq
    
    # Compute FFT of g
    G = fft(g) * dx  # Scale by dx for integration
    
    # Frequencies
    freqs = fftfreq(N, d=dx)
    
    # L1 and L2 norms of the Fourier transform
    L1_G = np.sum(np.abs(G)) * (2 * np.pi / N)  # Scale for frequency spacing
    L2_G = np.sqrt(np.sum(np.abs(G)**2)) * (2 * np.pi / N)
    
    if L2_G < 1e-15:
        return 1e10
    
    A_G = L1_G / L2_G
    
    # The upper bound is A(g) * A(G)
    bound = A_g * A_G
    
    return bound


def _objective(params):
    """Objective function for optimization: minimize the upper bound."""
    c0, c4, c8 = params
    try:
        return _upper_bound_C4(c0, c4, c8)
    except Exception:
        return 1e10


def construct():
    """
    Find three coefficients (c0, c4, c8) that minimize the upper bound
    on A(f) * A(Fourier(f)) using Nelder-Mead simplex optimization.
    """
    # Initial guess from the prior state of the art
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Run Nelder-Mead optimization
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
    
    if result.success or result.fun < _objective(x0):
        return np.array(result.x)
    else:
        # Fallback to initial guess if optimization fails
        return x0
# EVOLVE-BLOCK-END
