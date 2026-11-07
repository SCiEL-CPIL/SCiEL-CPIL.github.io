import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid basis:
    - First 7 params: Chebyshev polynomial basis (global structure)
    - Last 3 params: Radial basis functions (localized corrections)
    
    This combines the well-conditioned global approximation of Chebyshev polynomials
    with the localized flexibility of RBFs to capture nonlinear dynamics.
    """
    # Training ranges from problem description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    t_x = (2.0 * x - (x_max + x_min)) / x_range
    t_v = (2.0 * v - (v_max + v_min)) / v_range
    
    # Clip to [-1, 1] for stability
    t_x = np.clip(t_x, -1.0, 1.0)
    t_v = np.clip(t_v, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(t_x)
    T1_x = t_x
    T2_x = 2.0 * t_x * t_x - 1.0
    T3_x = 4.0 * t_x * t_x * t_x - 3.0 * t_x
    
    T0_v = np.ones_like(t_v)
    T1_v = t_v
    T2_v = 2.0 * t_v * t_v - 1.0
    T3_v = 4.0 * t_v * t_v * t_v - 3.0 * t_v
    
    # Basis functions for Chebyshev part (7 terms)
    # p0: Constant (bias)
    term0 = T0_x * T0_v
    
    # p1: Linear x
    term1 = T1_x * T0_v
    
    # p2: Linear v
    term2 = T0_x * T1_v
    
    # p3: Quadratic x
    term3 = T2_x * T0_v
    
    # p4: Quadratic v
    term4 = T0_x * T2_v
    
    # p5: Interaction x*v
    term5 = T1_x * T1_v
    
    # p6: Cubic x
    term6 = T3_x * T0_v
    
    # Radial basis functions for localized corrections (3 terms)
    # Each RBF is a Gaussian centered at (cx, cv) with width sigma
    # We use fixed centers spread across the domain and let BFGS fit the weights
    # Centers chosen to cover the training range
    
    # RBF 1: Center near origin
    cx1, cv1 = 0.0, 0.0
    sigma1 = 0.3
    r1 = (t_x - cx1)**2 + (t_v - cv1)**2
    term7 = np.exp(-r1 / (2.0 * sigma1**2))
    
    # RBF 2: Center at positive x, small v
    cx2, cv2 = 0.5, 0.2
    sigma2 = 0.25
    r2 = (t_x - cx2)**2 + (t_v - cv2)**2
    term8 = np.exp(-r2 / (2.0 * sigma2**2))
    
    # RBF 3: Center at negative x, small v
    cx3, cv3 = -0.5, -0.2
    sigma3 = 0.25
    r3 = (t_x - cx3)**2 + (t_v - cv3)**2
    term9 = np.exp(-r3 / (2.0 * sigma3**2))
    
    # Combine all terms with parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    pred = (p0 * term0 + 
            p1 * term1 + 
            p2 * term2 + 
            p3 * term3 + 
            p4 * term4 + 
            p5 * term5 + 
            p6 * term6 + 
            p7 * term7 + 
            p8 * term8 + 
            p9 * term9)
    
    return pred
