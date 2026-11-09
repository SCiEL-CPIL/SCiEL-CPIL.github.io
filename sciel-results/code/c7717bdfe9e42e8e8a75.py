"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using anisotropic RBF (Gaussian) kernels.
    
    Parameters (10 total):
    - params[0:5]: weights for 5 RBF kernels at fixed centers
    - params[5]: width scale for x (sigma_x)
    - params[6]: width scale for y (sigma_y)
    - params[7]: linear bias for x
    - params[8]: linear bias for y
    - params[9]: constant bias
    
    Kernel centers: (0,0), (0.5,0), (-0.5,0), (0,0.5), (0,-0.5)
    
    The RBF terms capture local structure while linear terms handle global trends.
    Widths are constrained to positive values via softplus-like transformation.
    """
    # Extract parameters
    w0, w1, w2, w3, w4 = params[0], params[1], params[2], params[3], params[4]
    sx_raw, sy_raw = params[5], params[6]
    bx, by, c = params[7], params[8], params[9]
    
    # Ensure positive widths: use exp to guarantee positivity
    # This makes the optimization landscape smoother for BFGS
    sx = np.exp(np.clip(sx_raw, -5, 5))  # sigma_x in [e^-5, e^5] ≈ [0.0067, 148]
    sy = np.exp(np.clip(sy_raw, -5, 5))  # sigma_y in [e^-5, e^5] ≈ [0.0067, 148]
    
    # Kernel centers
    cx = np.array([0.0, 0.5, -0.5, 0.0, 0.0])
    cy = np.array([0.0, 0.0, 0.0, 0.5, -0.5])
    
    # Compute anisotropic Gaussian RBF kernels
    # K_i(x,y) = exp(-((x-cx_i)^2/sx^2 + (y-cy_i)^2/sy^2))
    
    dx0 = x - cx[0]
    dy0 = y - cy[0]
    k0 = np.exp(-(dx0 * dx0) / (sx * sx) - (dy0 * dy0) / (sy * sy))
    
    dx1 = x - cx[1]
    dy1 = y - cy[1]
    k1 = np.exp(-(dx1 * dx1) / (sx * sx) - (dy1 * dy1) / (sy * sy))
    
    dx2 = x - cx[2]
    dy2 = y - cy[2]
    k2 = np.exp(-(dx2 * dx2) / (sx * sx) - (dy2 * dy2) / (sy * sy))
    
    dx3 = x - cx[3]
    dy3 = y - cy[3]
    k3 = np.exp(-(dx3 * dx3) / (sx * sx) - (dy3 * dy3) / (sy * sy))
    
    dx4 = x - cx[4]
    dy4 = y - cy[4]
    k4 = np.exp(-(dx4 * dx4) / (sx * sx) - (dy4 * dy4) / (sy * sy))
    
    # Linear terms for global trends
    lx = x
    ly = y
    
    # Combine: RBF weights * kernels + linear biases
    pred = (w0 * k0 + w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4 +
            bx * lx + by * ly + c)
    
    return pred
# EVOLVE-BLOCK-END
