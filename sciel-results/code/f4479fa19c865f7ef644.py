"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses physically-motivated basis: restoring, cubic nonlinearity, damping terms,
    and cross-coupling terms typical of nonlinear oscillators (Duffing + van der Pol type).
    """
    # Restoring forces (position-dependent)
    x_linear = x
    x_cubic = x ** 3
    x_quadratic = x ** 2
    
    # Damping forces (velocity-dependent)
    v_linear = v
    v_quadratic = v ** 2
    v_cubic = v ** 3
    
    # Cross-coupling terms
    xv = x * v
    xv2 = x * v ** 2
    x2v = x ** 2 * v
    
    # Build the 10-term model
    return (params[0] * x_linear +
            params[1] * x_cubic +
            params[2] * x_quadratic +
            params[3] * v_linear +
            params[4] * v_quadratic +
            params[5] * v_cubic +
            params[6] * xv +
            params[7] * xv2 +
            params[8] * x2v +
            params[9])
# EVOLVE-BLOCK-END
