"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Pure Chebyshev polynomial basis of degree 9
    # Numerically stable orthogonal basis on [-2, 2]
    # T_n(x/2) where T_n is the Chebyshev polynomial of the first kind
    
    t = x / 2.0  # scale to [-1, 1]
    
    # Compute Chebyshev polynomials T_0 through T_9
    # Using recurrence: T_0 = 1, T_1 = t, T_{n+1} = 2*t*T_n - T_{n-1}
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    T5 = 2 * t * T4 - T3
    T6 = 2 * t * T5 - T4
    T7 = 2 * t * T6 - T5
    T8 = 2 * t * T7 - T6
    T9 = 2 * t * T8 - T7
    
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
