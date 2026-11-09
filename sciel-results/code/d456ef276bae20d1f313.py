# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with physically-motivated basis.
    
    Combines polynomial state terms (Duffing stiffness, Van der Pol damping)
    with coupled time-state terms (parametric excitation) to capture the
    dynamics of a forced nonlinear oscillator.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions:
    p0: Constant offset
    p1: Linear position (x)
    p2: Linear velocity (v)
    p3: Quadratic position (x^2)
    p4: Cubic position (x^3) - Duffing stiffness
    p5: Quadratic velocity (v^2)
    p6: Cross term x^2*v - Van der Pol damping
    p7: Cross term x*v - state coupling
    p8: Coupled term t_c*x - parametric position coupling
    p9: Coupled term t_c*v - parametric velocity coupling
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center time for numerical stability (midpoint of [30, 50] is 40)
    t_c = t - 40.0
    
    # Construct basis functions
    f0 = np.ones_like(x)
    f1 = x
    f2 = v
    f3 = x ** 2
    f4 = x ** 3
    f5 = v ** 2
    f6 = x ** 2 * v
    f7 = x * v
    f8 = t_c * x
    f9 = t_c * v
    
    # Linear combination of basis functions
    return (p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 +
            p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9)
# EVOLVE-BLOCK-END
