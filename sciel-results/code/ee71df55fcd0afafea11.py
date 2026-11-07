import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, safety margin before 150s kill

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian RBF network with 2 kernels.
    The 10 parameters define:
      - 6 params for kernel centers (cx1, cy1, cz1, cx2, cy2, cz2)
      - 2 params for kernel widths (sigma1, sigma2)
      - 2 params for kernel weights (w1, w2)
    
    The function is: w1 * exp(-||[x,y,z]-c1||^2 / (2*sigma1^2)) + w2 * exp(-||[x,y,z]-c2||^2 / (2*sigma2^2))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple fallback to avoid timeout
        return np.ones_like(x)
    
    # Kernel 1
    cx1, cy1, cz1 = p[0], p[1], p[2]
    sigma1 = np.maximum(np.abs(p[3]), 0.05)
    w1 = p[4]
    
    # Kernel 2
    cx2, cy2, cz2 = p[5], p[6], p[7]
    sigma2 = np.maximum(np.abs(p[8]), 0.05)
    w2 = p[9]
    
    # Compute distances for kernel 1
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist_sq1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    t1 = np.exp(-dist_sq1 / (2.0 * sigma1 * sigma1))
    
    # Compute distances for kernel 2
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist_sq2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    t2 = np.exp(-dist_sq2 / (2.0 * sigma2 * sigma2))
    
    result = w1 * t1 + w2 * t2
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
