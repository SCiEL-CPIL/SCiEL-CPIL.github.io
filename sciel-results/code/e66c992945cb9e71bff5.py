"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: linear + cross-term polynomial basis + one 3D Gaussian RBF kernel.
    
    Parameters layout (10 total):
      - params[0]:   bias (b)
      - params[1]:   coefficient for x
      - params[2]:   coefficient for y
      - params[3]:   coefficient for z
      - params[4]:   coefficient for x*y
      - params[5]:   RBF center x (cx)
      - params[6]:   RBF center y (cy)
      - params[7]:   RBF center z (cz)
      - params[8]:   RBF width (sigma)
      - params[9]:   RBF weight (w)
    
    Output: b + a1*x + a2*y + a3*z + a4*x*y 
          + w * exp(-||[x,y,z] - c||^2 / (2*sigma^2))
    """
    # Extract parameters
    b = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    cx = params[5]
    cy = params[6]
    cz = params[7]
    sigma = params[8]
    w = params[9]
    
    # Polynomial part
    poly = b + a1 * x + a2 * y + a3 * z + a4 * x * y
    
    # RBF part
    dx = x - cx
    dy = y - cy
    dz = z - cz
    dist_sq = dx * dx + dy * dy + dz * dz
    
    # Clamp sigma to avoid division by zero or extremely small values
    eps = 1e-6
    s = np.maximum(np.abs(sigma), eps)
    s_sq = s * s
    
    # Gaussian kernel: exponent is always <= 0, so exp is stable
    k = np.exp(-dist_sq / (2.0 * s_sq))
    
    result = poly + w * k
    
    return result
# EVOLVE-BLOCK-END
