"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a basis of Chebyshev polynomials, trigonometric functions,
    and Gaussian bumps. This basis is designed to be well-conditioned for BFGS optimization
    while providing sufficient flexibility to approximate non-linear target functions.
    """
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    z = x / 2.0
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2 * z**2 - 1
    T3 = 4 * z**3 - 3 * z
    T4 = 8 * z**4 - 8 * z**2 + 1
    
    # Trigonometric functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Gaussian bumps centered at -1, 0, 1
    # Sigma chosen to be ~0.5-1.0 width in the original x scale
    g_1 = np.exp(-(x - 1.0)**2)
    g_0 = np.exp(-x**2)
    g_m1 = np.exp(-(x + 1.0)**2)
    
    # Combine basis functions with parameters
    # params is an array of length 10
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * sin_x +
         params[6] * cos_x +
         params[7] * g_1 +
         params[8] * g_0 +
         params[9] * g_m1)
         
    return y
# EVOLVE-BLOCK-END
