import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Parameters:
    # p[0]: weight for RBF center (0, 0)
    # p[1]: weight for RBF center (0.5, 0)
    # p[2]: weight for RBF center (-0.5, 0)
    # p[3]: weight for RBF center (0, 0.5)
    # p[4]: weight for RBF center (0, -0.5)
    # p[5]: log-width for x direction (sigma_x = exp(p[5]))
    # p[6]: log-width for y direction (sigma_y = exp(p[6]))
    # p[7]: constant bias
    # p[8]: linear x bias
    # p[9]: linear y bias
    
    # Extract parameters
    w0 = params[0]
    w1 = params[1]
    w2 = params[2]
    w3 = params[3]
    w4 = params[4]
    
    # Ensure positive widths using exp
    # Clip to avoid overflow/underflow issues in exp, though BFGS usually stays reasonable
    # exp(1) ~ 2.7, exp(-10) ~ 4e-5, exp(10) ~ 22000
    # If sigma is too small, exp(-large) becomes 0, which is fine.
    # If sigma is too large, it acts as a constant.
    sigma_x = np.exp(params[5])
    sigma_y = np.exp(params[6])
    
    # Calculate squared distances for each center
    # Center 0: (0, 0)
    dx0 = x - 0.0
    dy0 = y - 0.0
    r0_sq = (dx0 * dx0) / (sigma_x * sigma_x) + (dy0 * dy0) / (sigma_y * sigma_y)
    k0 = np.exp(-r0_sq)
    
    # Center 1: (0.5, 0)
    dx1 = x - 0.5
    dy1 = y - 0.0
    r1_sq = (dx1 * dx1) / (sigma_x * sigma_x) + (dy1 * dy1) / (sigma_y * sigma_y)
    k1 = np.exp(-r1_sq)
    
    # Center 2: (-0.5, 0)
    dx2 = x - (-0.5)
    dy2 = y - 0.0
    r2_sq = (dx2 * dx2) / (sigma_x * sigma_x) + (dy2 * dy2) / (sigma_y * sigma_y)
    k2 = np.exp(-r2_sq)
    
    # Center 3: (0, 0.5)
    dx3 = x - 0.0
    dy3 = y - 0.5
    r3_sq = (dx3 * dx3) / (sigma_x * sigma_x) + (dy3 * dy3) / (sigma_y * sigma_y)
    k3 = np.exp(-r3_sq)
    
    # Center 4: (0, -0.5)
    dx4 = x - 0.0
    dy4 = y - (-0.5)
    r4_sq = (dx4 * dx4) / (sigma_x * sigma_x) + (dy4 * dy4) / (sigma_y * sigma_y)
    k4 = np.exp(-r4_sq)
    
    # Linear biases
    b_const = params[7]
    b_x = params[8] * x
    b_y = params[9] * y
    
    # Combine
    result = (w0 * k0 +
              w1 * k1 +
              w2 * k2 +
              w3 * k3 +
              w4 * k4 +
              b_const +
              b_x +
              b_y)
              
    return result
