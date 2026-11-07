import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a separable tensor-product basis of Chebyshev polynomials.
    
    The basis consists of products T_i(x) * T_j(y) for i, j in {0, 1, 2}.
    This gives 9 basis functions. The 10th parameter is unused (or can be 
    interpreted as a global offset if we adjust the basis, but standard 
    Chebyshev T0=1 covers the constant term). 
    To strictly use 10 params, we can add T3(x)T0(y) or similar, but 
    typically the first 9 (up to degree 2 in both) are sufficient for 
    smooth functions. Let's stick to the 9 most relevant low-order terms 
    and set the 10th param to multiply a zero vector to avoid affecting 
    the fit, or better, include T2(x)T2(y) which is already there.
    
    Wait, the prompt says params is length 10. 
    Basis functions:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T0(x)T2(y) = 2y^2 - 1
    5: T1(x)T1(y) = xy
    6: T2(x)T1(y) = (2x^2-1)y
    7: T1(x)T2(y) = x(2y^2-1)
    8: T2(x)T2(y) = (2x^2-1)(2y^2-1)
    
    That's 9. The 10th parameter can be assigned to a higher order term 
    like T3(x)T0(y) = 4x^3 - 3x, or we can just pad with zeros if we 
    believe the function is low-order. However, adding a 10th basis 
    function is safer to utilize the parameter space. Let's add T3(x)T0(y).
    
    Actually, looking at the inspiration "multi-scale RBF" and "Chebyshev", 
    Chebyshev is generally superior for polynomial-like smooth functions 
    on [-1,1]. The current NN approach might be overfitting or struggling 
    with optimization landscape. Chebyshev basis is linear in params, 
    so BFGS will converge very quickly to the global minimum of the 
    linear least squares problem (or close to it).
    
    Note: The evaluator fits params. Since the model is linear in params, 
    the optimal solution is the least squares solution. BFGS on a quadratic 
    convex function converges rapidly.
    
    We must ensure numerical stability.
    """
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    # 0: 1
    # 1: x
    # 2: y
    # 3: 2x^2 - 1
    # 4: 2y^2 - 1
    # 5: xy
    # 6: (2x^2 - 1)y
    # 7: x(2y^2 - 1)
    # 8: (2x^2 - 1)(2y^2 - 1)
    # 9: (4x^3 - 3x)  # Adding a cubic term in x to utilize the 10th param
    
    # The evaluator expects the function to return predictions.
    # It fits params to minimize MSE.
    # prediction = sum(params[i] * basis_i(x,y))
    
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T0_x * T2_y
    b5 = T1_x * T1_y
    b6 = T2_x * T1_y
    b7 = T1_x * T2_y
    b8 = T2_x * T2_y
    b9 = T3_x * T0_y  # This is T3(x)
    
    # Combine linearly
    # params is shape (10,)
    # We need to return an array of shape (N,)
    
    # Ensure params is flattened if it comes in as a column vector, though 
    # the contract says it's an array of length 10.
    p = np.asarray(params).flatten()
    
    pred = (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
            
    return pred
