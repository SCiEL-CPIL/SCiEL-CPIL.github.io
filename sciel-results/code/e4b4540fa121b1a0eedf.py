"""New approach: Analytic reduction of the uncertainty product to a 1D L1 norm minimization.
Uses global optimization (Differential Evolution) to avoid local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import scipy.integrate


def _phys_hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


# Precompute Hermite polynomials on a fine grid for fast evaluation
_X_GRID = np.linspace(-10.0, 10.0, 20000)
_H0 = _phys_hermite(0, _X_GRID)
_H4 = _phys_hermite(4, _X_GRID)
_H8 = _phys_hermite(8, _X_GRID)
_H12 = _phys_hermite(12, _X_GRID)
_GAUSS = np.exp(-0.5 * _X_GRID * _X_GRID)
_DX = _X_GRID[1] - _X_GRID[0]

# H(0) values for the constraint
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 10395.0


def _objective(coeffs):
    """
    Minimizes the L1 norm of g(x).
    Based on the property that for Hermite-Gaussian functions, the L1 norm
    of the function and its Fourier transform are equal (up to a constant
    factor that doesn't affect the relative minimization), so minimizing
    A(g) minimizes A(g)A(g_hat).
    """
    a0, a4, a8 = coeffs
    # Calculate a12 such that g(0) = 0
    # g(0) = (a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0) * 1 = 0
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Construct g(x) on the grid
    # g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2)
    poly = a0 * _H0 + a4 * _H4 + a8 * _H8 + a12 * _H12
    g = poly * _GAUSS
    
    # L1 norm approximation
    l1_norm = np.sum(np.abs(g)) * _DX
    return l1_norm


def construct():
    """
    Uses Differential Evolution to find the global minimum of the L1 norm.
    Bounds are set generously to allow exploration.
    """
    # Initial seed from Goncalves et al. (2017) or similar known good point
    # a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    seed = np.array([-1.13, 0.04, 0.0003])
    
    # Define bounds for the differential evolution
    # The coefficients can vary, but usually stay within reasonable ranges
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    try:
        # Use Differential Evolution for global optimization
        # This is robust against local minima which L-BFGS might fall into
        result = scipy.optimize.differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=True  # Polish with L-BFGS at the end
        )
        
        if result.success:
            return result.x
        else:
            # If DE fails, return the best found or seed
            return result.x if np.all(np.isfinite(result.x)) else seed
            
    except Exception:
        # Fallback to seed if optimization fails
        return seed
# EVOLVE-BLOCK-END
