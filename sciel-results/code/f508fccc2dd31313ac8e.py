import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Duffing-like structure with
    state-dependent stiffness and damping, plus a small residual polynomial.
    
    a = -x * (p0 + p1 * x^2 + p2 * v) - v * (p3 + p4 * x) + p5 + p6*x + p7*v + p8*x^2 + p9*v^2
    """
    # Scale inputs to roughly [-1, 1] range to help BFGS
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    X = x / 0.55
    V = v / 0.25
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Duffing-like core: -x * (stiffness) - v * (damping)
    # Stiffness: p0 + p1 * X^2 (nonlinear spring)
    # Damping: p3 + p4 * X (state-dependent damping)
    stiffness = p0 + p1 * X**2 + p2 * V
    damping = p3 + p4 * X
    
    core = -X * stiffness - V * damping
    
    # Residual linear/quadratic terms to capture any remaining structure
    residual = p5 + p6 * X + p7 * V + p8 * X**2 + p9 * V**2
    
    return core + residual
