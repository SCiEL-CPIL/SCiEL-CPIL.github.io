import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression: Nonlinear Oscillator Acceleration.
    
    Uses a basis of Chebyshev polynomials for state variables (x, v) to ensure
    orthogonality and numerical stability for the BFGS optimizer, combined with
    Fourier terms for time dependence.
    
    Inputs:
    t: Time, range [30, 50]
    x: Position, range [-0.2056, 0.1749]
    v: Velocity, range [-0.3391, 0.3624]
    params: 10 fitted constants
    
    Basis Functions:
    0: 1
    1: T1(xn) = xn
    2: T1(vn) = vn
    3: T2(xn) = 2*xn^2 - 1
    4: T2(vn) = 2*vn^2 - 1
    5: T1(xn)*T1(vn) = xn*vn
    6: T3(xn) = 4*xn^3 - 3*xn
    7: T3(vn) = 4*vn^3 - 3*vn
    8: sin(t)
    9: cos(t)
    """
    # Deadline guard
    start_time = time.time()
    
    # Map x and v to [-1, 1] for Chebyshev polynomials
    # x range: [-0.2056, 0.1749] -> center -0.01535, half-range 0.19025
    # v range: [-0.3391, 0.3624] -> center 0.01165, half-range 0.35075
    
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    # Clip to avoid slight out-of-range issues if any, though inputs are in range
    xn = (x - x_center) / x_half
    vn = (v - v_center) / v_half
    
    # Ensure values are within [-1, 1] for stability, though they should be
    # np.clip is safe
    
    # Chebyshev Polynomials
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    b0 = 1.0
    b1 = xn
    b2 = vn
    b3 = 2.0 * xn * xn - 1.0
    b4 = 2.0 * vn * vn - 1.0
    b5 = xn * vn
    b6 = 4.0 * xn * xn * xn - 3.0 * xn
    b7 = 4.0 * vn * vn * vn - 3.0 * vn
    
    # Time terms
    # Using t directly. If the period is not 1, these might not align.
    # However, sin(t) and cos(t) are the most general low-frequency terms.
    b8 = np.sin(t)
    b9 = np.cos(t)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear combination
    pred = (p0 * b0 + 
            p1 * b1 + 
            p2 * b2 + 
            p3 * b3 + 
            p4 * b4 + 
            p5 * b5 + 
            p6 * b6 + 
            p7 * b7 + 
            p8 * b8 + 
            p9 * b9)
    
    return pred
