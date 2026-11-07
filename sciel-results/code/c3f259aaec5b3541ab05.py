"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    3-Kernel Gaussian RBF Network with shared width, no individual weights, and no bias.
    
    Parameters layout (10 total):
      - params[0]:   shared width (sigma)
      - params[1:4]: center of kernel 1 (cx1, cy1, cz1)
      - params[4:7]: center of kernel 2 (cx2, cy2, cz2)
      - params[7:10]: center of kernel 3 (cx3, cy3, cz3)
    
    Output: exp(-||[x,y,z] - c1||^2 / (2*sigma^2)) 
          + exp(-||[x,y,z] - c2||^2 / (2*sigma^2))
          + exp(-||[x,y,z] - c3||^2 / (2*sigma^2))
    """
    # Extract parameters
    sigma = params[0]
    cx1, cy1, cz1 = params[1], params[2], params[3]
    cx2, cy2, cz2 = params[4], params[5], params[6]
    cx3, cy3, cz3 = params[7], params[8], params[9]
    
    # Clamp sigma to avoid division by zero
    eps = 1e-6
    s = np.maximum(np.abs(sigma), eps)
    s_sq = s * s
    
    # Compute squared distances
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist1_sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist2_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    
    dx3 = x - cx3
    dy3 = y - cy3
    dz3 = z - cz3
    dist3_sq = dx3 * dx3 + dy3 * dy3 + dz3 * dz3
    
    # Compute Gaussian kernels
    k1 = np.exp(-dist1_sq / (2.0 * s_sq))
    k2 = np.exp(-dist2_sq / (2.0 * s_sq))
    k3 = np.exp(-dist3_sq / (2.0 * s_sq))
    
    # Sum of kernels
    result = k1 + k2 + k3
    
    return result
# EVOLVE-BLOCK-END
