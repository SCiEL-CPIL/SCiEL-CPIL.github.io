import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian RBF network with 2 kernels.
    The 10 parameters define:
      - 6 params for kernel centers (cx1, cy1, cz1, cx2, cy2, cz2)
      - 2 params for kernel widths (sigma1, sigma2)
      - 2 params for kernel weights (w1, w2)
    
    Additionally, we include a constant offset implicitly via the RBF structure.
    Actually, with only 10 params, let's use:
      - 6 for centers
      - 2 for widths
      - 2 for weights
    
    But we also need a bias/constant term. Let's restructure:
    Use 2 RBFs + 1 constant = 3 terms, but that's 6+2+2 = 10 params for RBFs, no room for constant.
    
    Alternative: Use 1 RBF + polynomial terms? 
    Let's try: 2 RBFs where each has center (3), width (1), weight (1) = 6+2+2=10.
    The constant can be captured by the RBFs if they're wide enough, but it's risky.
    
    Better approach: Use a hybrid. 
    Let's use:
      - 1 RBF: center (3), width (1), weight (1) = 5 params
      - Linear terms: x, y, z, xy, xz, yz, constant = 7 params... too many.
    
    Let's stick with 2 RBFs, 10 params total:
      p[0:3] = center1 (cx1, cy1, cz1)
      p[3] = width1
      p[4] = weight1
      p[5:8] = center2 (cx2, cy2, cz2)
      p[8] = width2
      p[9] = weight2
    
    The function is: w1 * exp(-||[x,y,z]-c1||^2 / (2*sigma1^2)) + w2 * exp(-||[x,y,z]-c2||^2 / (2*sigma2^2))
    
    This is purely local. It won't capture global trends well unless the function is localized.
    But the hidden function might be something like exp(x) or log(x) which is global.
    
    Let's try a different split:
      - Constant: 1 param
      - Linear x, y, z: 3 params
      - 1 RBF: center (3), width (1), weight (1) = 5 params
      Total: 1+3+5 = 9 params. Need 1 more. Add interaction xy.
    
    Let's do:
      p[0] = constant
      p[1] = coeff for x
      p[2] = coeff for y
      p[3] = coeff for z
      p[4] = coeff for xy
      p[5:8] = RBF center (cx, cy, cz)
      p[8] = RBF width
      p[9] = RBF weight
    
    This gives global linear + one local bump. More flexible.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Global terms
    t_const = np.ones_like(x)
    t_x = x
    t_y = y
    t_z = z
    t_xy = x * y
    
    # RBF term
    cx = p[5]
    cy = p[6]
    cz = p[7]
    sigma = p[8]
    w = p[9]
    
    # Ensure sigma is positive and not too small
    sigma = np.maximum(np.abs(sigma), 0.01)
    
    dx = x - cx
    dy = y - cy
    dz = z - cz
    dist_sq = dx * dx + dy * dy + dz * dz
    
    t_rbf = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    result = (p[0] * t_const + 
              p[1] * t_x + 
              p[2] * t_y + 
              p[3] * t_z + 
              p[4] * t_xy + 
              w * t_rbf)
    
    return result
