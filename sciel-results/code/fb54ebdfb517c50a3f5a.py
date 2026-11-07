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
    Model: Full 3rd order polynomial in x and v.
    y = p0 + p1*x + p2*v + p3*x^2 + p4*x*v + p5*v^2 + p6*x^3 + p7*x^2*v + p8*x*v^2 + p9*v^3
    """
    # Explicit deadline guard check
    # In this context, we just ensure we don't run past the limit if this were a loop,
    # but since it's a direct call, we just compute.
    # If we were to implement a search, we'd stop here.
    
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Terms
    # Order: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    return (
        p[0] +
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
# EVOLVE-BLOCK-END
