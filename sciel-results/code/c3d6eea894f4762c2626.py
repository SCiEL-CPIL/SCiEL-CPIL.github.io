"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a hybrid basis:
    - Fourier series in time for the driving force (captures periodicity)
    - Chebyshev polynomials in normalized position for the restoring force
    - Linear and nonlinear damping terms
    
    This structure exploits orthogonality to create a better-conditioned
    optimization landscape for BFGS.
    
    Features (10 params):
    0: bias
    1: T0(x_n) = 1 (constant restoring offset)
    2: T1(x_n) = x_n (linear restoring)
    3: T2(x_n) = 2x_n^2 - 1 (quadratic restoring)
    4: T3(x_n) = 4x_n^3 - 3x_n (cubic restoring, Duffing-like)
    5: v_n (linear damping)
    6: v_n * |v_n| (nonlinear damping)
    7: sin(t) (first harmonic drive)
    8: cos(t) (first harmonic drive, phase shift)
    9: sin(2t) (second harmonic drive)
    """
    # Normalize inputs to help BFGS converge from initial guess of 1s
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Clamp x_n to [-1, 1] for numerical stability of Chebyshev polynomials
    # This is a soft clamp using tanh to maintain differentiability
    x_c = np.tanh(x_n)
    
    # Chebyshev polynomials of the first kind
    # T0(x) = 1
    T0 = np.ones_like(x_c)
    # T1(x) = x
    T1 = x_c
    # T2(x) = 2x^2 - 1
    T2 = 2.0 * x_c * x_c - 1.0
    # T3(x) = 4x^3 - 3x
    T3 = 4.0 * x_c * x_c * x_c - 3.0 * x_c
    
    # Damping terms
    damping_linear = v_n
    damping_nonlinear = v_n * np.abs(v_n)
    
    # Fourier drive terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2.0 * t)
    
    # Assemble the equation
    result = (params[0] * T0 +
              params[1] * T1 +
              params[2] * T2 +
              params[3] * T3 +
              params[4] * damping_linear +
              params[5] * damping_nonlinear +
              params[6] * sin_t +
              params[7] * cos_t +
              params[8] * sin_2t)
    
    # Safety: ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
