import numpy as np

def equation(x, y, params):
    # Use Chebyshev polynomial basis up to total degree 3.
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    # Chebyshev polynomials for x
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    # Chebyshev polynomials for y
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Basis functions: Tensor products T_i(x) * T_j(y) where i+j <= 3
    # Degree 0:
    b0 = T0_x * T0_y  # 1
    
    # Degree 1:
    b1 = T1_x * T0_y  # x
    b2 = T0_x * T1_y  # y
    
    # Degree 2:
    b3 = T2_x * T0_y  # T2(x)
    b4 = T1_x * T1_y  # x*y
    b5 = T0_x * T2_y  # T2(y)
    
    # Degree 3:
    b6 = T3_x * T0_y  # T3(x)
    b7 = T2_x * T1_y  # T2(x)*y
    b8 = T1_x * T2_y  # x*T2(y)
    b9 = T0_x * T3_y  # T3(y)
    
    return (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
            params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
            params[8]*b8 + params[9]*b9)
