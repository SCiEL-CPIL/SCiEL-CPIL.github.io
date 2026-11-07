"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Physics-informed basis: Duffing + Van der Pol + Pendulum features.
    
    Basis functions:
    1. x          : Linear restoring force
    2. x^3        : Duffing nonlinearity
    3. v          : Linear damping
    4. v|v|       : Quadratic damping
    5. x^2 v      : Van der Pol damping
    6. x|v|       : Cross damping
    7. sin(x)     : Pendulum restoring
    8. sin(3x)    : Higher harmonic restoring
    9. v sin(x)   : Velocity-position coupling
    10. 1         : Bias
    """
    p = params
    
    # Precompute basis functions
    x2 = x * x
    x3 = x2 * x
    v_abs = np.abs(v)
    v_abs_v = v * v_abs  # v|v|
    x2_v = x2 * v
    x_v_abs = x * v_abs  # x|v|
    sin_x = np.sin(x)
    sin_3x = np.sin(3.0 * x)
    v_sin_x = v * sin_x
    
    return (
        p[0] * x +
        p[1] * x3 +
        p[2] * v +
        p[3] * v_abs_v +
        p[4] * x2_v +
        p[5] * x_v_abs +
        p[6] * sin_x +
        p[7] * sin_3x +
        p[8] * v_sin_x +
        p[9]
    )
# EVOLVE-BLOCK-END
