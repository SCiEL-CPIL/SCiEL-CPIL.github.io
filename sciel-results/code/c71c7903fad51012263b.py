"""Symbolic regression with single RBF + polynomial trend."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single 3D Gaussian RBF + linear + quadratic trend with 10 parameters.
    Params layout:
      0: cx   (center x of RBF)
      1: cy   (center y of RBF)
      2: cz   (center z of RBF)
      3: lw   (log-width of RBF; w = exp(lw))
      4: a    (RBF amplitude)
      5: bx   (linear x coefficient)
      6: by   (linear y coefficient)
      7: bz   (linear z coefficient)
      8: bxx  (quadratic x^2 coefficient, isotropic quadratic)
      9: c    (constant term)
    """
    p = params
    cx, cy, cz, lw, a = p[0], p[1], p[2], p[3], p[4]
    bx, by, bz, bxx, c = p[5], p[6], p[7], p[8], p[9]
    
    # Width in log-space ensures positivity and improves conditioning
    w = np.exp(lw)
    w2 = w * w
    
    dx = x - cx
    dy = y - cy
    dz = z - cz
    r_sq = dx * dx + dy * dy + dz * dz
    g = np.exp(-r_sq / (2.0 * w2))
    
    # Polynomial trend: linear + isotropic quadratic in x + constant
    poly = bx * x + by * y + bz * z + bxx * x * x + c
    
    return a * g + poly
# EVOLVE-BLOCK-END
