"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier-Exponential Hybrid Model.
    
    This approach separates the time-dependent periodic behavior (using 
    Fourier harmonics) from the state-dependent non-linear dynamics (using 
    hyperbolic tangent and polynomial terms). This avoids division 
    instabilities and captures saturation effects via tanh.
    
    Structure:
    y = Drive(t) + State(x, v)
    
    Drive(t) = p0*cos(t) + p1*sin(t) + p2*cos(2t) + p3*sin(2t)
    State(x, v) = p4*tanh(p5*x) + p6*tanh(p7*v) + p8*x*v + p9*x^2
    
    Params:
    p0-p3: Amplitudes for 1st and 2nd harmonic time components
    p4-p9: Coefficients for state-dependent non-linear terms
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Time-dependent drive with 2nd harmonic
    drive = p0 * np.cos(t) + p1 * np.sin(t) + p2 * np.cos(2 * t) + p3 * np.sin(2 * t)
    
    # State-dependent non-linear terms
    # Using tanh for bounded non-linearity (saturation)
    state_x = p4 * np.tanh(p5 * x)
    state_v = p6 * np.tanh(p7 * v)
    
    # Cross and quadratic terms for damping/stiffness
    state_cross = p8 * x * v
    state_quad = p9 * (x ** 2)
    
    return drive + state_x + state_v + state_cross + state_quad
# EVOLVE-BLOCK-END
