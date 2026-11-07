"""Symbolic regression with Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two 3D Gaussian RBF kernels with 10 parameters.
    Params layout:
      0: cx1  (center x of kernel 1)
      1: cy1  (center y of kernel 1)
      2: cz1  (center z of kernel 1)
      3: w1   (width of kernel 1)
      4: a1   (weight of kernel 1)
      5: cx2  (center x of kernel 2)
      6: cy2  (center y of kernel 2)
      7: cz2  (center z of kernel 2)
      8: w2   (width of kernel 2)
      9: a2   (weight of kernel 2)
    """
    p = params
    cx1, cy1, cz1, w1, a1 = p[0], p[1], p[2], p[3], p[4]
    cx2, cy2, cz2, w2, a2 = p[5], p[6], p[7], p[8], p[9]
    
    # Ensure positive widths to avoid division by zero
    w1 = np.maximum(w1, 1e-6)
    w2 = np.maximum(w2, 1e-6)
    
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    r1_sq = dx1**2 + dy1**2 + dz1**2
    g1 = np.exp(-r1_sq / (2.0 * w1**2))
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    r2_sq = dx2**2 + dy2**2 + dz2**2
    g2 = np.exp(-r2_sq / (2.0 * w2**2))
    
    return a1 * g1 + a2 * g2
# EVOLVE-BLOCK-END
