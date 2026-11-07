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
    Model: Scaled 3rd-order polynomial in raw x and v.
    p[0] is a global scale factor; p[1:10] are term coefficients.
    This structure helps BFGS converge from all-ones init by separating
    magnitude (p[0]) from shape (p[1:10]).
    """
    p = params
    
    # Precompute powers of raw inputs (already O(1) scale)
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Inner polynomial: shape of the oscillator response
    inner = (
        p[1] * x +
        p[2] * v +
        p[3] * x2 +
        p[4] * x * v +
        p[5] * v2 +
        p[6] * x3 +
        p[7] * x2 * v +
        p[8] * x * v2 +
        p[9] * v3
    )
    
    # Global scale: p[0] adjusts overall magnitude
    # Starting from all-ones, inner ~ O(1) for typical inputs,
    # so p[0] can quickly converge to ~0.1 to match target magnitude
    return p[0] * inner
# EVOLVE-BLOCK-END
