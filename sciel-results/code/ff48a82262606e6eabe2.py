# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-motivated model.
    
    Model: a = -k1*tanh(x/s1) - k2*x^3/s2^2 - c1*tanh(v/t1) - c2*v^3/t2^2
             + d1*x*v - d2*x^2*v - d3*v^2*x + f1*sin(w*x) + f2*cos(w*v) + bias
    
    This captures:
    - Nonlinear restoring force (tanh saturation + cubic term)
    - Nonlinear damping (tanh + cubic)
    - Velocity-dependent position coupling
    - Periodic forcing terms
    - Bias term
    
    Inputs are scaled to unit range for better conditioning.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to roughly [-1, 1]
    X = x * 1.8   # brings x to ~[-1.03, 0.95]
    V = v * 3.0   # brings v to ~[-0.53, 0.85]
    
    # Nonlinear restoring force: tanh saturation + cubic
    # tanh(X/s) provides softening/hardening depending on s
    s1 = np.maximum(np.abs(p[0]) * 0.1 + 0.1, 0.1)  # scale for tanh, keep positive
    s2 = np.maximum(np.abs(p[1]) * 0.1 + 0.1, 0.1)
    
    restoring = p[0] * np.tanh(X / s1) + p[1] * X * X * X / (s2 * s2)
    
    # Nonlinear damping: tanh + cubic
    t1 = np.maximum(np.abs(p[2]) * 0.1 + 0.1, 0.1)
    t2 = np.maximum(np.abs(p[3]) * 0.1 + 0.1, 0.1)
    
    damping = p[2] * np.tanh(V / t1) + p[3] * V * V * V / (t2 * t2)
    
    # Coupling terms
    coupling = p[4] * X * V + p[5] * X * X * V + p[6] * V * V * X
    
    # Periodic terms
    w = np.maximum(np.abs(p[7]) * 0.5 + 0.5, 0.5)
    periodic = p[7] * np.sin(w * X) + p[8] * np.cos(w * V)
    
    # Bias
    bias = p[9] * np.ones_like(X)
    
    # Combine: acceleration = -restoring - damping + coupling + periodic + bias
    y = -restoring - damping + coupling + periodic + bias
    
    return y
# EVOLVE-BLOCK-END
