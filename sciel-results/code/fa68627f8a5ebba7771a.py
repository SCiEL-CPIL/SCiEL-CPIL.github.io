"""Rewritten program using RBF-based local linearization for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using local linearization
    with Gaussian RBF centers on a 2D grid of (x, v).
    
    Each RBF multiplies a local linear term (1 + x + v) to capture non-additive
    coupling in the phase plane.
    """
    # Define fixed RBF centers on a 2D grid of (x, v)
    # Based on training ranges: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # Use 3x3 grid = 9 centers, but we only have 10 params total
    # Strategy: use 4 RBF centers, each with 2 params (one for the RBF weight, 
    # but that would be 8, plus 2 for global linear terms)
    # Actually, let's use a simpler approach:
    # We have 10 params. Let's use:
    # - 2 global linear terms: params[0] * x + params[1] * v
    # - 4 RBF centers, each contributing a local correction
    #   For each center, we use 1 param as the weight, and the local linear term is fixed
    #   That's 4 params for RBF weights
    # - Total so far: 2 + 4 = 6, leaving 4 more
    # Let's refine: use 5 RBF centers with 1 param each = 5, plus 2 global = 7, plus 3 more for nonlinear terms
    
    # Better approach: 
    # params[0]: coefficient for x (global linear)
    # params[1]: coefficient for v (global linear)
    # params[2:7]: weights for 5 RBF centers (each RBF multiplies (1 + a*x + b*v) locally)
    # params[7]: coefficient for x*v interaction
    # params[8]: coefficient for x^2
    # params[9]: coefficient for v^2
    
    # RBF centers (fixed, chosen to cover the training range)
    # x range: [-0.2056, 0.1749] -> center at ~-0.015, spread ~0.2
    # v range: [-0.3391, 0.3624] -> center at ~0.012, spread ~0.35
    
    # 5 RBF centers:
    # c1: (x=-0.1, v=-0.2)
    # c2: (x=-0.1, v=0.2)
    # c3: (x=0.1, v=-0.2)
    # c4: (x=0.1, v=0.2)
    # c5: (x=0.0, v=0.0)
    
    cx1, cv1 = -0.1, -0.2
    cx2, cv2 = -0.1, 0.2
    cx3, cv3 = 0.1, -0.2
    cx4, cv4 = 0.1, 0.2
    cx5, cv5 = 0.0, 0.0
    
    # RBF bandwidth
    sigma = 0.15
    
    # Compute RBF values
    rbf1 = np.exp(-((x - cx1)**2 + (v - cv1)**2) / (2 * sigma**2))
    rbf2 = np.exp(-((x - cx2)**2 + (v - cv2)**2) / (2 * sigma**2))
    rbf3 = np.exp(-((x - cx3)**2 + (v - cv3)**2) / (2 * sigma**2))
    rbf4 = np.exp(-((x - cx4)**2 + (v - cv4)**2) / (2 * sigma**2))
    rbf5 = np.exp(-((x - cx5)**2 + (v - cv5)**2) / (2 * sigma**2))
    
    # Local linear terms for each RBF: (1 + x + v)
    local1 = 1.0 + x + v
    local2 = 1.0 + x + v
    local3 = 1.0 + x + v
    local4 = 1.0 + x + v
    local5 = 1.0 + x + v
    
    # Global linear terms
    global_x = params[0] * x
    global_v = params[1] * v
    
    # RBF contributions
    rbf_contrib = (params[2] * rbf1 * local1 +
                   params[3] * rbf2 * local2 +
                   params[4] * rbf3 * local3 +
                   params[5] * rbf4 * local4 +
                   params[6] * rbf5 * local5)
    
    # Nonlinear interaction terms
    nonlinear = (params[7] * x * v +
                 params[8] * x**2 +
                 params[9] * v**2)
    
    return global_x + global_v + rbf_contrib + nonlinear
# EVOLVE-BLOCK-END
