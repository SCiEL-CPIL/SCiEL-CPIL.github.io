import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a tensor-product Chebyshev basis.
    Basis functions are products of Chebyshev polynomials T_k(x) and T_m(y).
    
    We use degrees 0, 1, 2 for both x and y (9 terms) plus T_3(x)T_3(y) (1 term).
    Total 10 terms.
    
    Chebyshev polynomials of the first kind:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    """
    # Compute Chebyshev polynomials for x
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    # Compute Chebyshev polynomials for y
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Construct the 10 basis functions
    # 3x3 grid of T_0..T_2
    phi = np.zeros_like(x)
    
    # Row 0: T0_x * [T0_y, T1_y, T2_y]
    phi += params[0] * T0_x * T0_y
    phi += params[1] * T0_x * T1_y
    phi += params[2] * T0_x * T2_y
    
    # Row 1: T1_x * [T0_y, T1_y, T2_y]
    phi += params[3] * T1_x * T0_y
    phi += params[4] * T1_x * T1_y
    phi += params[5] * T1_x * T2_y
    
    # Row 2: T2_x * [T0_y, T1_y, T2_y]
    phi += params[6] * T2_x * T0_y
    phi += params[7] * T2_x * T1_y
    phi += params[8] * T2_x * T2_y
    
    # Term 9: T3_x * T3_y (higher order interaction)
    phi += params[9] * T3_x * T3_y
    
    return phi
