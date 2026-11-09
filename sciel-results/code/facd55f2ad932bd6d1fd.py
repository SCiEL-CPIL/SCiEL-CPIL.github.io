"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis combining:
    - Low-degree polynomial terms for smooth global structure
    - Tanh-activated terms for capturing non-smooth/step-like features
    - Interaction terms with non-linear activations
    
    This is structurally different from pure polynomial bases because tanh
    functions can approximate piecewise-linear behavior and sharp transitions.
    
    10 basis terms:
    0: 1                          (bias)
    1: x                          (linear x)
    2: y                          (linear y)
    3: x*y                        (bilinear interaction)
    4: tanh(3*x)                  (saturated x)
    5: tanh(3*y)                  (saturated y)
    6: tanh(3*x) * tanh(3*y)      (saturated interaction)
    7: x^2 * y                    (quadratic-x times y)
    8: x * y^2                    (x times quadratic-y)
    9: tanh(5*(x+y))              (diagonal nonlinearity)
    """
    # Basic polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * y
    
    # Tanh-activated terms (capture saturation/step behavior)
    b4 = np.tanh(3.0 * x)
    b5 = np.tanh(3.0 * y)
    b6 = np.tanh(3.0 * x) * np.tanh(3.0 * y)
    
    # Higher-order polynomial interactions
    b7 = x * x * y
    b8 = x * y * y
    
    # Diagonal nonlinearity
    b9 = np.tanh(5.0 * (x + y))
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
