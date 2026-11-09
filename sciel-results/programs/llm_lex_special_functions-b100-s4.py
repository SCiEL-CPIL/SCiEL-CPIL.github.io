"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a normalized Chebyshev polynomial basis.
    
    params[0:10] are coefficients for the 10 Chebyshev polynomials T_0 through T_9.
    The basis functions are normalized by their RMS values over the domain [0.1, 6]
    to balance gradient magnitudes across all parameters, making the BFGS optimization
    landscape more isotropic.
    """
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev polynomials
    # t = (2*x - (a+b)) / (b-a) where a=0.1, b=6
    a, b = 0.1, 6.0
    t = (2.0 * x - (a + b)) / (b - a)
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    # T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Normalize each basis function by its RMS value over the domain [0.1, 6]
    # Use a fine grid to estimate RMS
    x_grid = np.linspace(a, b, 1000)
    t_grid = (2.0 * x_grid - (a + b)) / (b - a)
    
    T_grid = np.zeros((10, len(x_grid)))
    T_grid[0, :] = 1.0
    T_grid[1, :] = t_grid
    for n in range(2, 10):
        T_grid[n, :] = 2.0 * t_grid * T_grid[n-1, :] - T_grid[n-2, :]
    
    # RMS normalization: divide each row by its RMS value
    rms = np.sqrt(np.mean(T_grid ** 2, axis=1))
    # Avoid division by zero
    rms[rms < 1e-12] = 1.0
    T_norm = T / rms[:, np.newaxis]
    
    # Return the linear combination of normalized Chebyshev polynomials
    return T_norm.T @ params
# EVOLVE-BLOCK-END
