"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network with 2 kernels in 3D.
    
    Parameters layout (10 total):
      - params[0:3]: center of kernel 1 (cx1, cy1, cz1)
      - params[3]:   width of kernel 1 (sigma1)
      - params[4]:   weight of kernel 1 (w1)
      - params[5:8]: center of kernel 2 (cx2, cy2, cz2)
      - params[8]:   width of kernel 2 (sigma2)
      - params[9]:   weight of kernel 2 (w2)
    
    Output: w1 * exp(-||[x,y,z] - c1||^2 / (2*sigma1^2)) 
          + w2 * exp(-||[x,y,z] - c2||^2 / (2*sigma2^2))
    """
    # Extract parameters
    cx1, cy1, cz1 = params[0], params[1], params[2]
    sigma1 = params[3]
    w1 = params[4]
    cx2, cy2, cz2 = params[5], params[6], params[7]
    sigma2 = params[8]
    w2 = params[9]
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist1_sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist2_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    
    # Clamp sigmas to avoid division by zero or negative values
    # Use a small epsilon to keep things numerically stable
    eps = 1e-8
    s1 = np.maximum(np.abs(sigma1), eps)
    s2 = np.maximum(np.abs(sigma2), eps)
    
    # Compute Gaussian kernels
    k1 = np.exp(-dist1_sq / (2.0 * s1 * s1))
    k2 = np.exp(-dist2_sq / (2.0 * s2 * s2))
    
    # Weighted sum
    result = w1 * k1 + w2 * k2
    
    return result
# EVOLVE-BLOCK-END
