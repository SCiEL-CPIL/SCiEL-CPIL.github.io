import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator (Duffing/Van der Pol style).
    Structure: a = -k(x) * x - c(v) * v + f(x, v)
    
    Parameters:
    p0: Base stiffness (linear)
    p1: Nonlinear stiffness coefficient (Duffing term)
    p2: Stiffness modulation scale
    p3: Stiffness modulation width
    p4: Base damping (linear)
    p5: Nonlinear damping coefficient (Van der Pol term)
    p6: Damping modulation scale
    p7: Damping modulation width
    p8: Cross-coupling coefficient
    p9: Cross-coupling scale
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute x^2 and v^2 for reuse
    x2 = x * x
    v2 = v * v

    # Stiffness term: -x * (p0 + p1 * tanh(p2 * x^2 + p3 * v))
    # The tanh provides a smooth, bounded nonlinearity that captures Duffing-like behavior
    stiffness_mod = np.tanh(p2 * x2 + p3 * v)
    stiffness_force = -x * (p0 + p1 * stiffness_mod)

    # Damping term: -v * (p4 + p5 * tanh(p6 * v^2 + p7 * x))
    # The tanh provides a smooth, bounded nonlinearity that captures Van der Pol-like behavior
    damping_mod = np.tanh(p6 * v2 + p7 * x)
    damping_force = -v * (p4 + p5 * damping_mod)

    # Cross-coupling term: p8 * x * v * tanh(p9 * (x^2 + v^2))
    # This captures potential interactions between position and velocity
    coupling_mod = np.tanh(p9 * (x2 + v2))
    coupling_force = p8 * x * v * coupling_mod

    # Total acceleration
    result = stiffness_force + damping_force + coupling_force

    # Check deadline
    if time.time() > deadline:
        pass

    return result
