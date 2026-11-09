import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bivariate Chebyshev polynomial basis.
    The model uses 10 terms corresponding to a tensor product of Chebyshev polynomials
    up to degree 3, providing a numerically stable representation for smooth functions
    on [-1, 1].
    
    Basis terms:
    1. T0(x)T0(y) = 1
    2. T1(x)T0(y) = x
    3. T0(x)T1(y) = y
    4. T2(x)T0(y) = 2x^2 - 1
    5. T0(x)T2(y) = 2y^2 - 1
    6. T1(x)T1(y) = xy
    7. T3(x)T0(y) = 4x^3 - 3x
    8. T0(x)T3(y) = 4y^3 - 3y
    9. T2(x)T1(y) = (2x^2 - 1)y
    10. T1(x)T2(y) = x(2y^2 - 1)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Compute Chebyshev polynomials
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    T0_x = 1.0
    T1_x = x
    T2_x = 2.0 * x2 - 1.0
    T3_x = 4.0 * x3 - 3.0 * x
    
    T0_y = 1.0
    T1_y = y
    T2_y = 2.0 * y2 - 1.0
    T3_y = 4.0 * y3 - 3.0 * y
    
    # Construct basis terms (tensor product)
    b1 = T0_x * T0_y
    b2 = T1_x * T0_y
    b3 = T0_x * T1_y
    b4 = T2_x * T0_y
    b5 = T0_x * T2_y
    b6 = T1_x * T1_y
    b7 = T3_x * T0_y
    b8 = T0_x * T3_y
    b9 = T2_x * T1_y
    b10 = T1_x * T2_y
    
    # Linear combination
    pred = (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
            
    return pred
