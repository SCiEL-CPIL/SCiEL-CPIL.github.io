import numpy as np

# Normalization constants based on training range
# x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physically-motivated nonlinear oscillator model (Duffing-van der Pol style)
    with polynomial flexibility.
    
    Inputs:
    x: position
    v: velocity
    params: 10 coefficients
    
    Output:
    Prediction of the next state or derivative.
    """
    # Normalize inputs to [-1, 1] to improve conditioning
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Basis functions
    # 0: Constant
    t0 = np.ones_like(xn)
    # 1: Linear x
    t1 = xn
    # 2: Quadratic x
    t2 = xn2
    # 3: Cubic x (Duffing stiffness)
    t3 = xn3
    # 4: Linear v
    t4 = vn
    # 5: Cross term xv
    t5 = xn * vn
    # 6: Cross term x^2 v (Duffing-van der Pol damping part)
    t6 = xn2 * vn
    # 7: Cross term x^3 v
    t7 = xn3 * vn
    # 8: Quadratic v
    t8 = vn2
    # 9: Cubic v (van der Pol damping)
    t9 = vn3
    
    # Combine with parameters
    # params[0] * t0 + ...
    result = (params[0] * t0 + 
              params[1] * t1 + 
              params[2] * t2 + 
              params[3] * t3 + 
              params[4] * t4 + 
              params[5] * t5 + 
              params[6] * t6 + 
              params[7] * t7 + 
              params[8] * t8 + 
              params[9] * t9)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
