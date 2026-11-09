"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) approximation for a nonlinear oscillator.
    
    Uses 10 parameters to define 3 Gaussian basis functions in (x, v) space.
    Each basis function has:
      - center_x, center_v (2 params)
      - width (1 param)
      - weight (1 param)
    Total: 3 * 4 = 12, but we have only 10 params.
    
    Let's use 2 basis functions with 4 params each (8 params) plus 2 global params:
      - bias term (1 param)
      - linear damping coefficient (1 param)
    
    Actually, let's be more systematic. With 10 params:
    - 3 basis functions, each with 2 center coords + 1 width = 3 params each = 9 params
    - 1 weight shared? No, that's bad.
    
    Better: 2 Gaussian basis functions
      - Basis 1: center_x, center_v, width, weight (4 params)
      - Basis 2: center_x, center_v, width, weight (4 params)
      - Linear term: slope for x (1 param)
      - Linear term: slope for v (1 param)
    Total: 10 params
    
    This gives a flexible local approximation with some global linear structure.
    """
    # Parameters:
    # p0: center_x of basis 1
    # p1: center_v of basis 1
    # p2: width of basis 1
    # p3: weight of basis 1
    # p4: center_x of basis 2
    # p5: center_v of basis 2
    # p6: width of basis 2
    # p7: weight of basis 2
    # p8: linear coefficient for x
    # p9: linear coefficient for v
    
    cx1 = params[0]
    cv1 = params[1]
    w1 = params[2]
    wt1 = params[3]
    
    cx2 = params[4]
    cv2 = params[5]
    w2 = params[6]
    wt2 = params[7]
    
    lin_x = params[8]
    lin_v = params[9]
    
    # Ensure widths are positive to avoid numerical issues
    # Use softplus-like: width = abs(w) + 0.01 to keep positive and smooth
    w1_safe = np.abs(w1) + 0.01
    w2_safe = np.abs(w2) + 0.01
    
    # Gaussian basis functions
    # phi = exp(-((x-cx)^2 + (v-cv)^2) / (2*w^2))
    dx1 = x - cx1
    dv1 = v - cv1
    r1_sq = dx1 * dx1 + dv1 * dv1
    phi1 = np.exp(-r1_sq / (2.0 * w1_safe * w1_safe))
    
    dx2 = x - cx2
    dv2 = v - cv2
    r2_sq = dx2 * dx2 + dv2 * dv2
    phi2 = np.exp(-r2_sq / (2.0 * w2_safe * w2_safe))
    
    # Combine
    result = wt1 * phi1 + wt2 * phi2 + lin_x * x + lin_v * v
    
    return result
# EVOLVE-BLOCK-END
