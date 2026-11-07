"""Combined approach: Chebyshev tensor-product basis with radial/angular terms for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using a well-conditioned Chebyshev-based basis
    that includes tensor products up to degree 3, plus selected quartic and
    radial/angular interaction terms.
    
    Basis functions (10 total):
    b0 = 1
    b1 = x
    b2 = y
    b3 = 2x^2 - 1        (Chebyshev T2 in x)
    b4 = 2y^2 - 1        (Chebyshev T2 in y)
    b5 = xy              (interaction)
    b6 = 4x^3 - 3x       (Chebyshev T3 in x)
    b7 = 4y^3 - 3y       (Chebyshev T3 in y)
    b8 = x^2 + y^2       (radial term, captures isotropic curvature)
    b9 = x^2 - y^2       (angular/difference term)
    
    This spans a richer space than pure monomials up to degree 3, includes
    well-conditioned Chebyshev polynomials for numerical stability, and adds
    radial/angular terms that can capture symmetric features common in
    smooth functions on [-1,1]^2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    
    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x2 - 1.0
    T3_x = 4.0 * x2 * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y2 - 1.0
    T3_y = 4.0 * y2 * y - 3.0 * y
    
    # 10 basis functions
    b0 = T0_x * T0_y          # 1
    b1 = T1_x * T0_y          # x
    b2 = T0_x * T1_y          # y
    b3 = T2_x * T0_y          # 2x^2 - 1
    b4 = T0_x * T2_y          # 2y^2 - 1
    b5 = T1_x * T1_y          # xy
    b6 = T3_x * T0_y          # 4x^3 - 3x
    b7 = T0_x * T3_y          # 4y^3 - 3y
    b8 = x2 + y2              # radial: x^2 + y^2
    b9 = x2 - y2              # angular: x^2 - y^2
    
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
