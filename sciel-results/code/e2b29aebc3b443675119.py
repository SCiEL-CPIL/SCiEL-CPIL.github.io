"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Uses a physics-informed bounded nonlinear basis (sin, tanh, cubic) 
    to capture periodic restoring and saturation effects.
    
    Inputs:
        x: position
        v: velocity
        params: 10 fitted constants
    
    Returns:
        Acceleration prediction (one per row)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Basis functions:
    # 0: 1 (Bias)
    # 1: x (Linear restoring force)
    # 2: v (Linear damping)
    # 3: sin(x) (Nonlinear restoring, periodic)
    # 4: tanh(x) (Saturating restoring)
    # 5: x^3 (Duffing cubic nonlinearity)
    # 6: v^2 (Velocity-dependent damping)
    # 7: sin(v) (Velocity nonlinearity)
    # 8: x*v (Cross coupling)
    # 9: x^2 (Quadratic position correction)
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = np.sin(x)
    b4 = np.tanh(x)
    b5 = x3
    b6 = v2
    b7 = np.sin(v)
    b8 = x * v
    b9 = x2
    
    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
# EVOLVE-BLOCK-END
