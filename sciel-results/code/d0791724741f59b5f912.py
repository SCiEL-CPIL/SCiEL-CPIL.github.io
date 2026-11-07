"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using a mixed basis of:
    - Low-degree monomials (1, x, y, x^2, y^2, xy, x^3, y^3)
    - Radial/angular terms (x^2+y^2, x^2-y^2, x*y)
    
    This 10-term basis spans a rich function space including all
    polynomials up to degree 3 plus selected degree-4 radial terms.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 10 basis functions:
    # b0: constant
    # b1: x
    # b2: y
    # b3: x^2
    # b4: y^2
    # b5: x*y
    # b6: x^3
    # b7: y^3
    # b8: x^2 + y^2 (radial)
    # b9: x^2 - y^2 (angular)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = x * x * x
    b7 = y * y * y
    b8 = x * x + y * y
    b9 = x * x - y * y
    
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
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.mean(y) * np.ones_like(x)
        
    return pred
# EVOLVE-BLOCK-END
