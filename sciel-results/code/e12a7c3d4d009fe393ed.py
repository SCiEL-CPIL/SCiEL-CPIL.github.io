# EVOLVE-BLOCK-START
import numpy as np

# Fixed scaling constants (not fitted) to bring inputs to O(1) for BFGS conditioning
_X_SCALE = 1.0 / 0.5728  # ~1.746, maps x range to ~[-1, 0.92]
_V_SCALE = 1.0 / 0.2821  # ~3.545, maps v range to ~[-0.62, 1.0]


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses scaled polynomial basis for better BFGS conditioning.
    """
    xs = x * _X_SCALE
    vs = v * _V_SCALE
    
    # Scaled powers
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2
    vs2 = vs * vs
    vs3 = vs2 * vs
    vs4 = vs2 * vs2
    
    # Cross terms (scaled)
    xs2_vs = xs2 * vs
    xs_vs2 = xs * vs2
    
    # Basis: 10 terms
    # 0: bias
    # 1: xs
    # 2: xs^3
    # 3: xs^4
    # 4: vs
    # 5: vs^3
    # 6: vs^4
    # 7: xs^2 * vs
    # 8: xs * vs^2
    # 9: xs * vs (linear cross)
    
    return (params[0] +
            params[1] * xs +
            params[2] * xs3 +
            params[3] * xs4 +
            params[4] * vs +
            params[5] * vs3 +
            params[6] * vs4 +
            params[7] * xs2_vs +
            params[8] * xs_vs2 +
            params[9] * xs * vs)
# EVOLVE-BLOCK-END
