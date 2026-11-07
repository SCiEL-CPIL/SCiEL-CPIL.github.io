"""Symbolic regression with 1 input; local basis expansion with learned centers."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial expansion of degree 9 with 10 parameters.
    
    Uses Chebyshev polynomials T_0(x) through T_9(x) as basis functions.
    Chebyshev polynomials provide a stable, orthogonal-like basis on [-1, 1]
    that efficiently captures smooth functions and oscillations without
    the numerical instability of high-degree monomial polynomials.
    
    We map x from [-2, 2] to [-1, 1] for better conditioning.
    
    Basis functions (after mapping z = x/2):
    0: T_0(z) = 1
    1: T_1(z) = z
    2: T_2(z) = 2z^2 - 1
    3: T_3(z) = 4z^3 - 3z
    ...up to T_9(z)
    """
    p = params
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    z = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence
    T0 = np.ones_like(z)
    T1 = z.copy()
    
    # Recurrence: T_{n+1}(z) = 2z * T_n(z) - T_{n-1}(z)
    T = [T0, T1]
    for n in range(2, 10):
        Tn = 2.0 * z * T[n-1] - T[n-2]
        T.append(Tn)
    
    y = (p[0] * T[0] +
         p[1] * T[1] +
         p[2] * T[2] +
         p[3] * T[3] +
         p[4] * T[4] +
         p[5] * T[5] +
         p[6] * T[6] +
         p[7] * T[7] +
         p[8] * T[8] +
         p[9] * T[9])
    
    return y
# EVOLVE-BLOCK-END
