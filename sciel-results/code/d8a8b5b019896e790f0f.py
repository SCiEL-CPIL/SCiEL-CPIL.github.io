"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters.
    Combines polynomial (up to degree 3) and trigonometric terms to approximate
    a wide range of smooth functions on [-1, 1]^2.
    
    The evaluator fits params via BFGS. We use a diverse set of 10 basis
    functions that cover linear, quadratic, cubic, and basic trigonometric behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 10 diverse basis functions:
    # 1. Constant
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. x^3
    # 8. y^3
    # 9. sin(x)
    # 10. sin(y)
    
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * x
    b5 = y * y
    b6 = x * y
    b7 = x * x * x
    b8 = y * y * y
    b9 = np.sin(x)
    b10 = np.sin(y)
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
# EVOLVE-BLOCK-END
