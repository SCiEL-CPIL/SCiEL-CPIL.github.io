import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled cubic polynomial basis for nonlinear oscillator.
    Scaling improves the conditioning of the design matrix for BFGS,
    allowing for more reliable convergence to the optimal parameters.
    
    Basis functions (10 total):
    1. 1
    2. x_s
    3. v_s
    4. x_s^2
    5. v_s^2
    6. x_s * v_s
    7. x_s^3
    8. v_s^3
    9. x_s^2 * v_s
    10. x_s * v_s^2
    
    Where x_s and v_s are scaled inputs.
    """
    # Scaling factors to normalize inputs to roughly [-1, 1]
    # x range approx [-0.57, 0.53], center ~0, half-range ~0.55
    # v range approx [-0.17, 0.28], center ~0.05, half-range ~0.22
    # Using simple scaling by max abs value to keep it simple and robust
    scale_x = 0.6
    scale_v = 0.3
    
    x_s = x / scale_x
    v_s = v / scale_v
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x_s
    b2 = v_s
    b3 = x_s * x_s
    b4 = v_s * v_s
    b5 = x_s * v_s
    b6 = b3 * x_s
    b7 = b4 * v_s
    b8 = b3 * v_s
    b9 = b4 * x_s
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
