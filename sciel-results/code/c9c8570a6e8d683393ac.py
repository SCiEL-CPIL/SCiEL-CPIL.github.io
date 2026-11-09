import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomials of the first kind on [-1, 1]
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct prediction using 10 parameters
    # This basis spans polynomials up to degree 3 in x and y
    # Using orthogonal basis improves conditioning for BFGS
    pred = (params[0] * T0_x * T0_y +
            params[1] * T1_x * T0_y +
            params[2] * T0_x * T1_y +
            params[3] * T2_x * T0_y +
            params[4] * T0_x * T2_y +
            params[5] * T1_x * T1_y +
            params[6] * T3_x * T0_y +
            params[7] * T0_x * T3_y +
            params[8] * T2_x * T1_y +
            params[9] * T1_x * T2_y)
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to linear if something went wrong (should not happen)
        pred = params[0] + params[1] * x + params[2] * y
        if not np.all(np.isfinite(pred)):
            pred = np.zeros_like(x)
            
    return pred
