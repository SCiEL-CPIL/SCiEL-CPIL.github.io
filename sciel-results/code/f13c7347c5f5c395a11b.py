"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# The evaluator calls equation(t, x, v, params).
# We define a robust nonlinear oscillator model.
# Structure: Fourier series in t + Polynomial in x and v.
# This covers Duffing (x^3), Van der Pol (x^2*v, v^3), and general periodic driving.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    y = A(t) + B(x, v)
    
    A(t) = p0*sin(t) + p1*cos(t)
    B(x, v) = p2*x + p3*x^2 + p4*x^3 + p5*v + p6*x*v + p7*x^2*v + p8*v^2 + p9*v^3
    
    Total 10 parameters.
    """
    # Pre-compute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    # Combine with parameters
    # p0, p1: Driving force (Fourier series, 1st harmonics)
    # p2, p3, p4: Spring force (Linear, Quadratic, Cubic)
    # p5: Linear damping
    # p6, p7, p8, p9: Nonlinear damping terms (x*v, x^2*v, v^2, v^3)
    
    y = (params[0] * sin_t + params[1] * cos_t +
         params[2] * x + params[3] * x2 + params[4] * x3 +
         params[5] * v + params[6] * xv + params[7] * x2v +
         params[8] * v2 + params[9] * v3)
         
    return y
# EVOLVE-BLOCK-END
