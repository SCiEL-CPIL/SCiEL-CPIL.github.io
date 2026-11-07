# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs (x, y) drawn from [-1, 1]^2.
    
    Strategy: Gaussian RBF network with learnable centers, widths, and weights.
    - 2 params for RBF center: (cx, cy)
    - 2 params for RBF width: (sx, sy) - inverse standard deviations
    - 1 param for RBF amplitude
    - 5 params for a polynomial basis: [1, x, y, x^2, y^2]
    
    The RBF term captures localized, non-linear features while the polynomial
    basis handles global trends. This is structurally different from Fourier+MLP.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # RBF center and width parameters
    cx = p[0]
    cy = p[1]
    sx = np.abs(p[2]) + 0.1  # ensure positive width
    sy = np.abs(p[3]) + 0.1  # ensure positive width
    
    # Compute RBF activation: exp(-((x-cx)^2/sx^2 + (y-cy)^2/sy^2))
    dx = x - cx
    dy = y - cy
    rbf_arg = -(dx**2 / (sx**2) + dy**2 / (sy**2))
    rbf = np.exp(rbf_arg)
    
    # RBF amplitude
    rbf_weight = p[4]
    rbf_part = rbf_weight * rbf
    
    # Polynomial basis: 1, x, y, x^2, y^2
    poly_part = (p[5] + 
                 p[6] * x + 
                 p[7] * y + 
                 p[8] * x**2 + 
                 p[9] * y**2)
    
    return rbf_part + poly_part
# EVOLVE-BLOCK-END
