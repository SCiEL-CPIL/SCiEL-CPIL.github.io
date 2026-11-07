"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    The model uses a physically motivated basis for a Duffing-like oscillator
    with damping and potential asymmetry.
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Model structure:
        p0 * x        (linear stiffness)
        p1 * v        (linear damping)
        p2 * x^2      (quadratic stiffness / asymmetry)
        p3 * x^3      (cubic stiffness / Duffing term)
        p4 * x*v      (velocity-dependent stiffness)
        p5 * v^2      (quadratic damping / drag)
        p6 * x^4      (quartic stiffness)
        p7 * x^2*v    (higher order damping)
        p8 * v^3      (cubic damping)
        p9            (constant bias)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and cross terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    return (
        p0 * x +
        p1 * v +
        p2 * x2 +
        p3 * x3 +
        p4 * xv +
        p5 * v2 +
        p6 * x4 +
        p7 * x2v +
        p8 * v3 +
        p9
    )

# EVOLVE-BLOCK-END
