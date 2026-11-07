"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Calculate Chebyshev polynomials of the first kind T_0 ... T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    T6 = 32.0 * t**6 - 48.0 * t**4 + 18.0 * t**2 - 1.0
    T7 = 64.0 * t**7 - 112.0 * t**5 + 56.0 * t**3 - 7.0 * t
    T8 = 128.0 * t**8 - 256.0 * t**6 + 160.0 * t**4 - 32.0 * t**2 + 1.0
    T9 = 256.0 * t**9 - 576.0 * t**7 + 432.0 * t**5 - 120.0 * t**3 + 9.0 * t
    
    # Linear combination of the basis functions
    # This is linear in params, ensuring robust convergence of the optimizer
    return (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            params[5] * T5 +
            params[6] * T6 +
            params[7] * T7 +
            params[8] * T8 +
            params[9] * T9)
# EVOLVE-BLOCK-END
