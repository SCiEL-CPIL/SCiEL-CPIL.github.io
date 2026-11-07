import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a full tensor-product Chebyshev basis.
    
    Uses T_i(x) * T_j(y) for i,j in {0,1,2} (9 terms) plus T3(x)*T1(y)
    as the 10th term to capture cubic interaction. This complete tensor
    product structure is well-conditioned on [-1,1]^2 and allows BFGS
    to converge reliably to the global least-squares minimum.
    
    Basis:
    0: T0(x)*T0(y) = 1
    1: T1(x)*T0(y) = x
    2: T0(x)*T1(y) = y
    3: T2(x)*T0(y) = 2x^2 - 1
    4: T0(x)*T2(y) = 2y^2 - 1
    5: T1(x)*T1(y) = x*y
    6: T2(x)*T1(y)
    7: T1(x)*T2(y)
    8: T2(x)*T2(y)
    9: T3(x)*T1(y)
    """
    # Chebyshev Polynomials of the First Kind
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    T0 = np.ones_like(x)
    T1_x = x
    T1_y = y
    
    T2_x = 2.0 * x * x - 1.0
    T2_y = 2.0 * y * y - 1.0
    
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    # Full tensor-product basis T_i(x) * T_j(y) for i,j in {0,1,2}
    # plus T3(x)*T1(y) as the 10th term
    b0 = T0 * T0                          # 1
    b1 = T1_x * T0                        # x
    b2 = T0 * T1_y                        # y
    b3 = T2_x * T0                        # 2x^2 - 1
    b4 = T0 * T2_y                        # 2y^2 - 1
    b5 = T1_x * T1_y                      # x*y
    b6 = T2_x * T1_y                      # (2x^2-1)*y
    b7 = T1_x * T2_y                      # x*(2y^2-1)
    b8 = T2_x * T2_y                      # (2x^2-1)*(2y^2-1)
    b9 = T3_x * T1_y                      # (4x^3-3x)*y
    
    # Linear combination of basis functions
    f = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
         
    return f
