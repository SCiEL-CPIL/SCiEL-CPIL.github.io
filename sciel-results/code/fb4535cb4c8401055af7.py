"""Symbolic regression for a nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration.
    
    Uses 10 parameters to define a sum of smooth Gaussian bumps in the (x, v) state space,
    with a linear time-dependent driving term.
    
    Parameters:
        t: time (array)
        x: position (array)
        v: velocity (array)
        params: 10-element array
            p0: center_x for bump 1
            p1: center_v for bump 1
            p2: center_x for bump 2
            p3: center_v for bump 2
            p4: center_x for bump 3
            p5: center_v for bump 3
            p6: common scale (inverse bandwidth)
            p7: weight for bump 1
            p8: weight for bump 2
            p9: weight for bump 3
    
    The model is:
        a = p7 * exp(-p6^2 * ((x-p0)^2 + (v-p1)^2)) 
          + p8 * exp(-p6^2 * ((x-p2)^2 + (v-p3)^2))
          + p9 * exp(-p6^2 * ((x-p4)^2 + (v-p5)^2))
    
    This captures arbitrary nonlinear acceleration fields in the (x, v) state space.
    """
    # Extract parameters
    cx1, cv1 = params[0], params[1]
    cx2, cv2 = params[2], params[3]
    cx3, cv3 = params[4], params[5]
    scale = params[6]
    w1, w2, w3 = params[7], params[8], params[9]
    
    # Ensure scale is positive for numerical stability
    scale = np.maximum(scale, 0.01)
    
    # Compute squared distances
    dx1 = x - cx1
    dv1 = v - cv1
    dx2 = x - cx2
    dv2 = v - cv2
    dx3 = x - cx3
    dv3 = v - cv3
    
    # Compute Gaussian bumps
    r1_sq = dx1**2 + dv1**2
    r2_sq = dx2**2 + dv2**2
    r3_sq = dx3**2 + dv3**2
    
    # Sum of weighted Gaussians
    result = (w1 * np.exp(-scale**2 * r1_sq) +
              w2 * np.exp(-scale**2 * r2_sq) +
              w3 * np.exp(-scale**2 * r3_sq))
    
    return result
# EVOLVE-BLOCK-END
