# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Duffing-like polynomial basis
    with input scaling for better numerical conditioning of BFGS.
    
    The model captures:
    - Linear and cubic restoring forces (Duffing oscillator)
    - Linear and cubic damping
    - Cross-coupling terms between position and velocity
    - Higher-order corrections for nonlinearity
    
    Inputs are scaled to roughly [-1, 1] range to improve optimizer conditioning.
    10 parameters are fitted via BFGS.
    """
    p = params
    
    # Scale inputs to improve numerical conditioning for BFGS
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    X = x / 0.6
    V = v / 0.3
    
    # Precompute powers for efficiency
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Basis functions:
    # Restoring force terms (odd in X)
    # b0: linear restoring
    # b1: cubic restoring (Duffing)
    # b2: quintic restoring (higher order)
    # b3: X * tanh(X) - smooth nonlinear restoring
    b0 = X
    b1 = X3
    b2 = X2 * X2 * X  # X^5
    b3 = X * np.tanh(X * 2.0)
    
    # Damping terms (odd in V)
    # b4: linear damping
    # b5: cubic damping
    # b6: quintic damping
    # b7: V * tanh(V) - smooth nonlinear damping
    b4 = V
    b5 = V3
    b6 = V2 * V2 * V  # V^5
    b7 = V * np.tanh(V * 3.0)
    
    # Cross-coupling terms
    # b8: X*V (linear coupling)
    # b9: X^2*V (asymmetric damping)
    b8 = X * V
    b9 = X2 * V
    
    # Combine with 10 parameters
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
