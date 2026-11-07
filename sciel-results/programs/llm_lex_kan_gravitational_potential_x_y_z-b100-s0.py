# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 2 kernels in 3D + bias.
    10 params: [w1, w2, c1x, c1y, c1z, c2x, c2y, c2z, s, bias]
    """
    p = params
    w1, w2 = p[0], p[1]
    c1x, c1y, c1z = p[2], p[3], p[4]
    c2x, c2y, c2z = p[5], p[6], p[7]
    s = p[8]
    bias = p[9]
    
    # Smooth positive width with bounded range for better BFGS behavior
    # Use tanh to keep width in a reasonable range: [0.1, 10]
    s_safe = 0.1 + 9.9 * (0.5 + 0.5 * np.tanh(s))
    
    # Compute squared distances
    d1_sq = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    
    # Gaussian kernels with stable computation
    inv_2s2 = 1.0 / (2.0 * s_safe**2)
    k1 = np.exp(-d1_sq * inv_2s2)
    k2 = np.exp(-d2_sq * inv_2s2)
    
    return w1 * k1 + w2 * k2 + bias
# EVOLVE-BLOCK-END
