"""Symbolic regression with composite basis: polynomial + trigonometric + tanh."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter composite basis for symbolic regression.
    
    The basis combines:
    - 4 Chebyshev polynomials T_0..T_3 (for smooth polynomial structure)
    - 3 trigonometric terms: sin(x), cos(x), sin(2x) (for periodic features)
    - 3 tanh-based terms: tanh(x), tanh(2x), tanh(0.5x) (for saturation/nonlinear features)
    
    All parameters are linear coefficients, making the optimization convex
    (linear least squares), which BFGS can solve reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure x is 1D
    x = x.ravel()
    
    # Build the 10-column basis matrix
    n = len(x)
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # Columns 0-3: Chebyshev polynomials T_0..T_3
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = 2.0 * x ** 2 - 1.0
    basis[:, 3] = 4.0 * x ** 3 - 3.0 * x
    
    # Columns 4-6: Trigonometric terms
    basis[:, 4] = np.sin(x)
    basis[:, 5] = np.cos(x)
    basis[:, 6] = np.sin(2.0 * x)
    
    # Columns 7-9: Tanh-based terms
    basis[:, 7] = np.tanh(x)
    basis[:, 8] = np.tanh(2.0 * x)
    basis[:, 9] = np.tanh(0.5 * x)
    
    # Prediction is linear combination of basis functions
    return basis @ params
