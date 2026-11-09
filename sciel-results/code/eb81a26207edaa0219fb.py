"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # params: p0..p9 (length 10)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Neural ODE style structure:
    # Term 1: Velocity-dependent term with saturation
    term_v = p0 * np.tanh(v)
    
    # Term 2: Position-dependent stiffness with nonlinear modulation
    # Stiffness k(x, v) = p1 + p2 * tanh(p3 * x**2 + p4 * v)
    stiffness = p1 + p2 * np.tanh(p3 * x**2 + p4 * v)
    term_x = -x * stiffness
    
    # Additional flexible terms using remaining parameters to cover linear/polynomial bases
    # This allows the model to approximate linear oscillators if tanh arguments are small
    # or capture specific nonlinear features like cross-coupling and quadratic damping.
    extra = p5 + p6 * x + p7 * v + p8 * x * v + p9 * v**2

    acc = term_v + term_x + extra

    return acc
# EVOLVE-BLOCK-END
