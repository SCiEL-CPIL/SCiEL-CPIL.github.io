"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for nonlinear oscillator acceleration using
    a sum of Gaussian RBFs in the (x, v) plane.
    
    The 10 parameters are organized as:
    - params[0:2]: center (cx1, cv1) of first Gaussian
    - params[2:4]: center (cx2, cv2) of second Gaussian
    - params[4:6]: center (cx3, cv3) of third Gaussian
    - params[6:8]: center (cx4, cv4) of fourth Gaussian
    - params[8:10]: widths (sigma1, sigma2) for all Gaussians
    
    y = sum_k exp(-((x-cx_k)^2 + (v-cv_k)^2) / (2*sigma^2))
    
    This allows localized nonlinear dynamics that global polynomials miss.
    """
    # Extract parameters
    cx1, cv1 = params[0], params[1]
    cx2, cv2 = params[2], params[3]
    cx3, cv3 = params[4], params[5]
    cx4, cv4 = params[6], params[7]
    sigma1 = params[8]
    sigma2 = params[9]
    
    # Ensure widths are positive and numerically stable
    sigma1 = np.maximum(sigma1, 0.01)
    sigma2 = np.maximum(sigma2, 0.01)
    
    # Precompute squared distances for each Gaussian center
    dx1 = x - cx1
    dy1 = v - cv1
    dx2 = x - cx2
    dy2 = v - cv2
    dx3 = x - cx3
    dy3 = v - cv3
    dx4 = x - cx4
    dy4 = v - cv4
    
    # Compute Gaussian terms with shared width structure
    # Use two different widths to capture multi-scale dynamics
    g1 = np.exp(-(dx1 * dx1 + dy1 * dy1) / (2.0 * sigma1 * sigma1))
    g2 = np.exp(-(dx2 * dx2 + dy2 * dy2) / (2.0 * sigma2 * sigma2))
    g3 = np.exp(-(dx3 * dx3 + dy3 * dy3) / (2.0 * sigma1 * sigma1))
    g4 = np.exp(-(dx4 * dx4 + dy4 * dy4) / (2.0 * sigma2 * sigma2))
    
    # Sum of Gaussians
    return g1 + g2 + g3 + g4
# EVOLVE-BLOCK-END
