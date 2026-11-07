"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model using a 2D Chebyshev polynomial basis.
    This improves conditioning over raw monomials and allows BFGS to converge reliably.
    Inputs are scaled to [-1, 1] based on training data ranges.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale x and v to [-1, 1]
    # Training ranges: x in [-0.3391, 0.3624], v in [-0.2056, 0.1749]
    x_mid = (0.3624 - 0.3391) / 2 + (-0.3391)  # 0.01165
    x_range = (0.3624 - 0.3391) / 2             # 0.35075
    
    v_mid = (0.1749 - 0.2056) / 2 + (-0.2056)   # -0.01535
    v_range = (0.1749 - 0.2056) / 2             # 0.19025

    # Avoid division by zero if range is 0 (should not happen here)
    x_scaled = (x - x_mid) / x_range
    v_scaled = (v - v_mid) / v_range

    # Compute Chebyshev polynomials T_n(z)
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    
    T0_x = np.ones_like(x_scaled)
    T1_x = x_scaled
    T2_x = 2 * x_scaled**2 - 1
    T3_x = 4 * x_scaled**3 - 3 * x_scaled

    T0_v = np.ones_like(v_scaled)
    T1_v = v_scaled
    T2_v = 2 * v_scaled**2 - 1
    T3_v = 4 * v_scaled**3 - 3 * v_scaled

    # Construct the 10 terms:
    # 4 position-only terms (T_0..T_3 in x, T_0 in v)
    # 3 velocity-only terms (T_0 in x, T_1..T_3 in v)
    # 3 interaction terms (T_1*T_1, T_1*T_2, T_2*T_1)
    
    term0 = T0_x * T0_v
    term1 = T1_x * T0_v
    term2 = T2_x * T0_v
    term3 = T3_x * T0_v
    term4 = T0_x * T1_v
    term5 = T0_x * T2_v
    term6 = T0_x * T3_v
    term7 = T1_x * T1_v
    term8 = T1_x * T2_v
    term9 = T2_x * T1_v

    return (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 +
            p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 +
            p8 * term8 + p9 * term9)
# EVOLVE-BLOCK-END
