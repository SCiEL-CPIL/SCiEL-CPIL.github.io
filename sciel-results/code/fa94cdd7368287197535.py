import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomials of the first kind
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Tensor product basis, 10 terms
    # Order:
    # 0: T0_x T0_y = 1
    # 1: T1_x T0_y = x
    # 2: T0_x T1_y = y
    # 3: T2_x T0_y = 2x^2 - 1
    # 4: T0_x T2_y = 2y^2 - 1
    # 5: T1_x T1_y = xy
    # 6: T3_x T0_y = 4x^3 - 3x
    # 7: T0_x T3_y = 4y^3 - 3y
    # 8: T2_x T1_y = (2x^2 - 1)y
    # 9: T1_x T2_y = x(2y^2 - 1)
    
    f = (params[0] * T0_x * T0_y +
         params[1] * T1_x * T0_y +
         params[2] * T0_x * T1_y +
         params[3] * T2_x * T0_y +
         params[4] * T0_x * T2_y +
         params[5] * T1_x * T1_y +
         params[6] * T3_x * T0_y +
         params[7] * T0_x * T3_y +
         params[8] * T2_x * T1_y +
         params[9] * T1_x * T2_y)
         
    return f
