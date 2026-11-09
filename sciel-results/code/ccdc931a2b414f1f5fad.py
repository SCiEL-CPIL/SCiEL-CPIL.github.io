"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural change: use a richer, more flexible basis that combines
    # multiple types of functions to capture diverse target shapes.
    #
    # Basis design (10 parameters total):
    # - 3 Gaussian bumps at different centers and widths for local features
    # - 3 Chebyshev polynomial terms for global smooth structure
    # - 2 sinusoidal terms with different frequencies for oscillatory features
    # - 1 linear term for overall trend
    # - 1 constant term for offset
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Gaussian basis functions with different centers and widths
    # Center 1: -1.0, width 0.6
    g1 = np.exp(-((x - (-1.0)) ** 2) / (2 * 0.6 ** 2))
    # Center 2: 0.0, width 0.5
    g2 = np.exp(-((x - 0.0) ** 2) / (2 * 0.5 ** 2))
    # Center 3: 1.0, width 0.6
    g3 = np.exp(-((x - 1.0) ** 2) / (2 * 0.6 ** 2))
    
    # Chebyshev polynomials on scaled domain [-1, 1]
    t = x / 2.0
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t ** 2 - 1
    
    # Sinusoidal terms with different frequencies
    sin1 = np.sin(2.0 * np.pi * x / 4.0)   # period 4
    cos1 = np.cos(2.0 * np.pi * x / 4.0)
    
    # Linear term
    lin = x
    
    # Constant term
    const = np.ones_like(x)
    
    # Combine all basis terms with the 10 parameters
    # params[0:3] for Gaussians, params[3:6] for Chebyshev,
    # params[6:8] for sinusoidal, params[8] for linear, params[9] for constant
    result = (g1 * params[0] +
              g2 * params[1] +
              g3 * params[2] +
              T0 * params[3] +
              T1 * params[4] +
              T2 * params[5] +
              sin1 * params[6] +
              cos1 * params[7] +
              lin * params[8] +
              const * params[9])
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
