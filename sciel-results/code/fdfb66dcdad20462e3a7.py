import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    
    Uses a rich set of basis functions (polynomial, trigonometric, interaction)
    to approximate the hidden function. The model is linear in the 10 parameters,
    which allows BFGS to converge quickly to the global minimum.
    
    Basis functions:
    1. 1 (bias)
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. sin(pi*x)
    8. cos(pi*y)
    9. sin(pi*(x+y))
    10. x^2*y
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure 1D arrays
    x = x.ravel()
    y = y.ravel()
    
    # Basis functions
    b1 = np.ones_like(x)          # bias
    b2 = x                        # linear x
    b3 = y                        # linear y
    b4 = x * x                    # quadratic x
    b5 = y * y                    # quadratic y
    b6 = x * y                    # interaction
    b7 = np.sin(np.pi * x)        # trig x
    b8 = np.cos(np.pi * y)        # trig y
    b9 = np.sin(np.pi * (x + y))  # trig interaction
    b10 = x * x * y               # cubic interaction
    
    # Linear combination
    pred = (p[0] * b1 +
            p[1] * b2 +
            p[2] * b3 +
            p[3] * b4 +
            p[4] * b5 +
            p[5] * b6 +
            p[6] * b7 +
            p[7] * b8 +
            p[8] * b9 +
            p[9] * b10)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
