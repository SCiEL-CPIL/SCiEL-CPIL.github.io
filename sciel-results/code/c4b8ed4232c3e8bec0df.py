# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Centered polynomial basis with interaction terms.
    
    Inputs are centered around 1.25 (midpoint of [0.5, 2]) to improve
    Hessian conditioning during BFGS optimization.
    
    Basis functions (10 total, linear in params):
    0: 1 (bias)
    1: cx
    2: cy
    3: cz
    4: cx * cy
    5: cy * cz
    6: cz * cx
    7: cx * cy * cz
    8: cx^2
    9: cy^2
    
    Since the model is linear in params, BFGS converges to the global
    optimum, avoiding local minima issues of nonlinear parameterizations.
    """
    p = params
    
    # Center inputs around midpoint of [0.5, 2]
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25
    
    # Build basis
    b0 = np.ones_like(cx)
    b1 = cx
    b2 = cy
    b3 = cz
    b4 = cx * cy
    b5 = cy * cz
    b6 = cz * cx
    b7 = cx * cy * cz
    b8 = cx * cx
    b9 = cy * cy
    
    # Linear combination
    out = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
           p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
           p[8] * b8 + p[9] * b9)
    
    return out
# EVOLVE-BLOCK-END
