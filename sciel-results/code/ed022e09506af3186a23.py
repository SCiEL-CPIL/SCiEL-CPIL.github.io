import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with separated odd-order
    restoring and damping forces, plus a bias and a linear cross-term.
    
    This approach enforces physical symmetries (odd powers for force/damping)
    which reduces the effective parameter space and improves conditioning.
    """
    # Scale inputs to improve numerical conditioning for the optimizer.
    # Based on training ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
    # Using 0.5 and 0.3 as characteristic scales.
    X = x / 0.5
    V = v / 0.3
    
    # Precompute odd powers for restoring force (x)
    X3 = X * X * X
    X5 = X3 * X * X
    X7 = X5 * X * X
    
    # Precompute odd powers for damping force (v)
    V3 = V * V * V
    V5 = V3 * V * V
    V7 = V5 * V * V
    
    # Cross term
    XV = X * V
    
    # Construct the expression
    # Restoring Force (Odd in X)
    # p0: X, p1: X^3, p2: X^5, p3: X^7
    R = params[0] * X + params[1] * X3 + params[2] * X5 + params[3] * X7
    
    # Damping Force (Odd in V)
    # p4: V, p5: V^3, p6: V^5, p7: V^7
    D = params[4] * V + params[5] * V3 + params[6] * V5 + params[7] * V7
    
    # Bias and Cross-term
    # p8: Constant
    # p9: X*V
    B = params[8]
    C = params[9] * XV
    
    return R + D + B + C
