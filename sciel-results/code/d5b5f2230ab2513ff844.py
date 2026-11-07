"""Symbolic regression: equation for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a nonlinear oscillator.
    
    Uses a Chebyshev polynomial basis in normalized coordinates.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a well-conditioned
    basis for fitting, which helps BFGS converge more reliably than monomial powers.
    
    Basis terms (10 total, total degree <= 3):
    1.  T_0(x)T_0(v)
    2.  T_0(x)T_1(v)
    3.  T_0(x)T_2(v)
    4.  T_0(x)T_3(v)
    5.  T_1(x)T_0(v)
    6.  T_1(x)T_1(v)
    7.  T_1(x)T_2(v)
    8.  T_2(x)T_0(v)
    9.  T_2(x)T_1(v)
    10. T_3(x)T_0(v)
    
    where T_n is the Chebyshev polynomial of the first kind.
    Inputs are normalized to roughly [-1, 1].
    """
    # Normalize inputs to O(1) scale based on training range max abs values
    # x range: [-0.5728, 0.5275] -> max abs ~ 0.57
    # v range: [-0.1754, 0.2821] -> max abs ~ 0.28
    # Using slightly wider scaling to ensure values stay within [-1, 1] mostly
    x_s = x / 0.58
    v_s = v / 0.29
    
    # Compute Chebyshev polynomials T_0, T_1, T_2, T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    x_T0 = np.ones_like(x_s)
    x_T1 = x_s
    x_T2 = 2.0 * x_s * x_s - 1.0
    x_T3 = 4.0 * x_s * x_s * x_s - 3.0 * x_s
    
    v_T0 = np.ones_like(v_s)
    v_T1 = v_s
    v_T2 = 2.0 * v_s * v_s - 1.0
    v_T3 = 4.0 * v_s * v_s * v_s - 3.0 * v_s
    
    # Construct the 10 basis terms
    b0 = x_T0 * v_T0
    b1 = x_T0 * v_T1
    b2 = x_T0 * v_T2
    b3 = x_T0 * v_T3
    b4 = x_T1 * v_T0
    b5 = x_T1 * v_T1
    b6 = x_T1 * v_T2
    b7 = x_T2 * v_T0
    b8 = x_T2 * v_T1
    b9 = x_T3 * v_T0
    
    # Linear combination of basis functions
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
