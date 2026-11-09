"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    Chebyshev polynomials provide better numerical conditioning for optimization
    compared to raw monomials, especially when starting from an arbitrary point.
    """
    # Scale inputs to [-1, 1] domain for Chebyshev
    X = x / 0.6
    V = v / 0.3

    # Clip to [-1, 1] to avoid numerical issues if inputs slightly out of range,
    # though training data is in range.
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Chebyshev polynomials for X
    T0_X = 1.0
    T1_X = X
    T2_X = 2.0 * X**2 - 1.0
    T3_X = 4.0 * X**3 - 3.0 * X

    # Chebyshev polynomials for V
    T0_V = 1.0
    T1_V = V
    T2_V = 2.0 * V**2 - 1.0
    T3_V = 4.0 * V**3 - 3.0 * V

    # Select 10 terms. We prioritize low-order terms which capture the main dynamics.
    # Order: Constant, X, V, X^2, XV, X^3, V^2, X^2V, XV^2, V^3
    # Note: This is a heuristic selection. 
    # Let's stick to the standard monomial set but mapped through Chebyshev?
    # Actually, just using the Chebyshev polynomials themselves as basis functions is fine.
    # Let's use:
    # 0: T0_X * T0_V (1)
    # 1: T1_X * T0_V (X)
    # 2: T0_X * T1_V (V)
    # 3: T2_X * T0_V (X^2)
    # 4: T1_X * T1_V (XV)
    # 5: T3_X * T0_V (X^3)
    # 6: T0_X * T2_V (V^2)
    # 7: T2_X * T1_V (X^2 V)
    # 8: T1_X * T2_V (X V^2)
    # 9: T0_X * T3_V (V^3)
    
    return (p0 * (T0_X * T0_V) +
            p1 * (T1_X * T0_V) +
            p2 * (T0_X * T1_V) +
            p3 * (T2_X * T0_V) +
            p4 * (T1_X * T1_V) +
            p5 * (T3_X * T0_V) +
            p6 * (T0_X * T2_V) +
            p7 * (T2_X * T1_V) +
            p8 * (T1_X * T2_V) +
            p9 * (T0_X * T3_V))
# EVOLVE-BLOCK-END
