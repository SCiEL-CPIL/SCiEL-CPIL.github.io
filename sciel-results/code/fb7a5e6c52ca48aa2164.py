import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product basis of Chebyshev
    polynomials up to degree 2 in each variable (3x3 = 9 terms) plus one
    global constant-free interaction term.
    
    The basis spans the space of functions that are separable sums of products
    of 1D Chebyshev polynomials: sum_{i=0}^{2} sum_{j=0}^{2} c_{ij} T_i(x) T_j(y).
    This is a rank-3 tensor product structure that captures all interactions
    up to quadratic order in each variable independently.
    
    The 10 basis terms are:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T0(x)T2(y) = 2y^2 - 1
    5: T1(x)T1(y) = x*y
    6: T2(x)T1(y) = (2x^2-1)*y
    7: T1(x)T2(y) = x*(2y^2-1)
    8: T2(x)T2(y) = (2x^2-1)*(2y^2-1)
    9: sin(x)*sin(y)  -- captures oscillatory interaction not in polynomial span
    
    Chebyshev polynomials are well-conditioned on [-1,1] and provide good
    approximation properties for smooth functions.
    """
    # Chebyshev polynomials of the first kind up to degree 2
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    
    # Construct the 10 basis functions
    b0 = T0_x * T0_y          # 1
    b1 = T1_x * T0_y          # x
    b2 = T0_x * T1_y          # y
    b3 = T2_x * T0_y          # 2x^2 - 1
    b4 = T0_x * T2_y          # 2y^2 - 1
    b5 = T1_x * T1_y          # x*y
    b6 = T2_x * T1_y          # (2x^2-1)*y
    b7 = T1_x * T2_y          # x*(2y^2-1)
    b8 = T2_x * T2_y          # (2x^2-1)*(2y^2-1)
    b9 = np.sin(x) * np.sin(y)  # oscillatory interaction
    
    # Linear combination
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
