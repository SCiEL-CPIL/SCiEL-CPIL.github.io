# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis of bounded nonlinearities.
    Inputs:
        x: position array
        v: velocity array
        params: array of 10 fitted constants
    Returns:
        Acceleration array (same shape as x)
    """
    # Unpack parameters for clarity
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear basis functions
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Construct the model:
    # - Linear terms in x and v (basic oscillator dynamics)
    # - Nonlinear restoring force approximations via sin(x) and tanh(x)
    # - Nonlinear damping approximations via sin(v) and tanh(v)
    # - Interaction term x*v for coupling
    # - Constant bias term
    
    # p0: linear position (restoring force)
    # p1: nonlinear position via sin(x)
    # p2: nonlinear position via tanh(x)
    # p3: linear velocity (damping)
    # p4: nonlinear velocity via sin(v)
    # p5: nonlinear velocity via tanh(v)
    # p6: interaction x*v
    # p7: x^2 (quadratic position, captures asymmetry)
    # p8: v^2 (quadratic velocity, captures nonlinear drag)
    # p9: bias/constant driving term
    
    x2 = x * x
    v2 = v * v
    xv = x * v
    
    result = (
        p0 * x +
        p1 * sin_x +
        p2 * tanh_x +
        p3 * v +
        p4 * sin_v +
        p5 * tanh_v +
        p6 * xv +
        p7 * x2 +
        p8 * v2 +
        p9
    )
    
    return result
# EVOLVE-BLOCK-END
