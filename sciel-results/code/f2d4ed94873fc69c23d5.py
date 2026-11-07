import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product Chebyshev basis.
    
    Uses T_i(x) * T_j(y) for i,j in {0,1,2}, giving 9 basis functions.
    The 10th parameter is used as a global scale factor to improve conditioning.
    This structure explicitly models the function as a sum of products of 1D functions,
    reducing the effective dimensionality of the optimization landscape.
    """
    # Calculate Chebyshev polynomials T0 to T2 for both variables
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0

    # Construct the 9 separable basis functions (3x3 grid)
    # Order: (i,j) where i is x-degree, j is y-degree
    f00 = T0_x * T0_y
    f10 = T1_x * T0_y
    f20 = T2_x * T0_y
    f01 = T0_x * T1_y
    f11 = T1_x * T1_y
    f21 = T2_x * T1_y
    f02 = T0_x * T2_y
    f12 = T1_x * T2_y
    f22 = T2_x * T2_y

    # Stack the 9 basis functions
    basis = np.column_stack((
        f00, f10, f20, f01, f11, f21, f02, f12, f22
    ))
    
    # Extract parameters: first 9 for basis, 10th as scale
    p = params[:9]
    scale = params[9]
    
    # Compute prediction with scale factor
    prediction = scale * (basis @ p)
    
    return prediction
