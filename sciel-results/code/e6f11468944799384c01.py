import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Global Chebyshev-Tensor Basis.
    
    This approach uses a fixed basis of 10 Chebyshev polynomial terms to approximate 
    the target function. The basis is designed to be well-conditioned and to capture 
    global trends and interactions up to degree 3.
    
    Basis functions:
    1.  1
    2.  x
    3.  y
    4.  2x^2 - 1
    5.  2y^2 - 1
    6.  xy
    7.  4x^3 - 3x
    8.  4y^3 - 3y
    9.  (2x^2 - 1)y
    10. x(2y^2 - 1)
    
    The model is linear in the parameters, so BFGS will quickly find the optimal 
    coefficients for this fixed basis. This is structurally different from the 
    RBF approach, which relies on local kernels.
    """
    start_time = time.time()
    
    # Ensure params length is 10
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # Compute Chebyshev polynomials
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    T1_x = x
    T1_y = y
    
    T2_x = 2.0 * x**2 - 1.0
    T2_y = 2.0 * y**2 - 1.0
    
    T3_x = 4.0 * x**3 - 3.0 * x
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis features
    # We want to maximize expressivity for smooth functions.
    # 1. Constant
    f1 = np.ones_like(x)
    
    # 2-3. Linear
    f2 = T1_x
    f3 = T1_y
    
    # 4-5. Quadratic pure
    f4 = T2_x
    f5 = T2_y
    
    # 6. Interaction linear-linear
    f6 = T1_x * T1_y
    
    # 7-8. Cubic pure
    f7 = T3_x
    f8 = T3_y
    
    # 9-10. Mixed quadratic-linear (degree 3 interactions)
    f9 = T2_x * T1_y
    f10 = T1_x * T2_y
    
    # Stack into a feature matrix (N, 10)
    # Note: We could also use f9 = T2_x * T2_y but that's degree 4, 
    # sticking to degree <=3 keeps the basis compact and stable.
    
    features = np.column_stack((
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
    ))
    
    # Prediction: features @ p
    pred = features @ p
    
    # Safety check for time (though this is very fast)
    if time.time() - start_time > 110:
        # Fallback to simple linear if something goes wrong
        return p[0] + p[1] * x + p[2] * y
        
    return pred
