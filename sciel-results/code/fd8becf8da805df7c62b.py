import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid Chebyshev-polynomial and trigonometric basis.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: T2(x) = 2x^2 - 1
    4: T2(y) = 2y^2 - 1
    5: x*y
    6: sin(x)
    7: sin(y)
    8: sin(x+y)
    9: sin(x-y)
    
    Using Chebyshev polynomials for the quadratic terms improves the conditioning
    of the basis matrix, which can help the BFGS optimizer converge more reliably.
    """
    # Polynomial terms (Chebyshev-like)
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    t2_x = 2.0 * x2 - 1.0
    t2_y = 2.0 * y2 - 1.0

    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    sin_xy_plus = np.sin(x + y)
    sin_xy_minus = np.sin(x - y)

    # Construct prediction
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * t2_x 
            + params[4] * t2_y 
            + params[5] * xy 
            + params[6] * sin_x 
            + params[7] * sin_y 
            + params[8] * sin_xy_plus 
            + params[9] * sin_xy_minus)
            
    return pred
