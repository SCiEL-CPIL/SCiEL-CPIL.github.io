"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a hybrid basis: Chebyshev polynomials (T0..T7) plus
    two non-polynomial basis functions (sin, gaussian) for richer expressiveness.
    """
    # Scale input to [-1, 1] for better conditioning
    u = x / 3.0

    # Chebyshev polynomials of the first kind, T0 through T7
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2.0 * u * T1 - T0
    T3 = 2.0 * u * T2 - T1
    T4 = 2.0 * u * T3 - T2
    T5 = 2.0 * u * T4 - T3
    T6 = 2.0 * u * T5 - T4
    T7 = 2.0 * u * T6 - T5

    # Non-polynomial basis functions
    # sin(2*pi*u) captures periodicity with period 1 in u (period 6 in x)
    sin_term = np.sin(2.0 * np.pi * u)
    # Gaussian bump centered at 0, width controlled by u^2
    gauss_term = np.exp(-2.0 * u * u)

    # Linear combination of all 10 basis functions
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * T6 +
         params[7] * T7 +
         params[8] * sin_term +
         params[9] * gauss_term)

    return y
# EVOLVE-BLOCK-END
