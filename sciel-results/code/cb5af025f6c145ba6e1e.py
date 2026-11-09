"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses tanh-based nonlinearities for bounded, smooth features.
    Features:
    1. 1
    2. x
    3. v
    4. tanh(5*x)
    5. tanh(5*v)
    6. tanh(5*x^2)
    7. tanh(5*v^2)
    8. tanh(5*x^3)
    9. tanh(5*v^3)
    10. tanh(5*x*v)
    """
    # Compute scaled inputs for tanh to ensure non-linearity is visible
    # Scaling factor 5 is chosen to make arguments O(1-2) for typical inputs
    sx = 5.0 * x
    sv = 5.0 * v
    sx2 = 5.0 * x * x
    sv2 = 5.0 * v * v
    sx3 = 5.0 * x * x * x
    sv3 = 5.0 * v * v * v
    sxv = 5.0 * x * v

    t1 = np.ones_like(x)
    t2 = x
    t3 = v
    t4 = np.tanh(sx)
    t5 = np.tanh(sv)
    t6 = np.tanh(sx2)
    t7 = np.tanh(sv2)
    t8 = np.tanh(sx3)
    t9 = np.tanh(sv3)
    t10 = np.tanh(sxv)

    return (params[0] * t1 +
            params[1] * t2 +
            params[2] * t3 +
            params[3] * t4 +
            params[4] * t5 +
            params[5] * t6 +
            params[6] * t7 +
            params[7] * t8 +
            params[8] * t9 +
            params[9] * t10)
# EVOLVE-BLOCK-END
