import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Chebyshev polynomial basis up to degree 3
    in two variables, with 10 parameters.
    
    Chebyshev polynomials of the first kind:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    
    Basis functions:
    0: T_0(x)T_0(y) = 1
    1: T_1(x)T_0(y) = x
    2: T_0(x)T_1(y) = y
    3: T_2(x)T_0(y) = 2x^2 - 1
    4: T_0(x)T_2(y) = 2y^2 - 1
    5: T_1(x)T_1(y) = xy
    6: T_3(x)T_0(y) = 4x^3 - 3x
    7: T_0(x)T_3(y) = 4y^3 - 3y
    8: T_2(x)T_1(y) = (2x^2 - 1)y
    9: T_1(x)T_2(y) = x(2y^2 - 1)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p = params
    
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Construct the linear combination of basis terms
    # This basis is orthogonal over the square [-1,1]x[-1,1] with respect to the weight 1/sqrt(1-x^2)1/sqrt(1-y^2)
    # which helps BFGS convergence compared to monomials.
    
    pred = (p[0] * T0_x * T0_y + 
            p[1] * T1_x * T0_y + 
            p[2] * T0_x * T1_y + 
            p[3] * T2_x * T0_y + 
            p[4] * T0_x * T2_y + 
            p[5] * T1_x * T1_y + 
            p[6] * T3_x * T0_y + 
            p[7] * T0_x * T3_y + 
            p[8] * T2_x * T1_y + 
            p[9] * T1_x * T2_y)
            
    return pred
