"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    
    Strategy: Hybrid Global-Local Basis.
    - Global Polynomial Terms (Bias, Linear, Quadratic): Captures smooth, global trends.
    - Local Gaussian Terms (Center, Min-Corner, Max-Corner): Captures localized features, 
      peaks, or non-linearities that polynomials struggle with, especially near boundaries.
    
    Parameters (10 total):
    p[0]: Bias
    p[1]: Coefficient for x
    p[2]: Coefficient for y
    p[3]: Coefficient for z
    p[4]: Coefficient for x^2
    p[5]: Coefficient for y^2
    p[6]: Coefficient for z^2
    p[7]: Weight for Gaussian centered at (1.0, 1.0, 1.0)
    p[8]: Weight for Gaussian centered at (0.5, 0.5, 0.5)
    p[9]: Weight for Gaussian centered at (2.0, 2.0, 2.0)
    
    The Gaussians use a fixed width (sigma=0.5, so exp(-2*d^2) or exp(-4*d^2)?)
    Let's use exp(-4 * d^2) for sharper localization.
    """
    p = params
    
    # Global Polynomial Basis
    # Centering might help conditioning, but the optimizer (BFGS) should handle it.
    # We keep raw x,y,z for simplicity and direct interpretation of linear terms.
    term_bias = p[0]
    term_x = p[1] * x
    term_y = p[2] * y
    term_z = p[3] * z
    term_x2 = p[4] * x**2
    term_y2 = p[5] * y**2
    term_z2 = p[6] * z**2
    
    # Local Gaussian Basis
    # Centers:
    # 1. Center of domain (1, 1, 1)
    # 2. Min corner (0.5, 0.5, 0.5)
    # 3. Max corner (2.0, 2.0, 2.0)
    
    # Distance squared from center (1,1,1)
    d_center = (x - 1.0)**2 + (y - 1.0)**2 + (z - 1.0)**2
    g_center = np.exp(-4.0 * d_center)
    
    # Distance squared from min corner (0.5,0.5,0.5)
    d_min = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
    g_min = np.exp(-4.0 * d_min)
    
    # Distance squared from max corner (2.0,2.0,2.0)
    d_max = (x - 2.0)**2 + (y - 2.0)**2 + (z - 2.0)**2
    g_max = np.exp(-4.0 * d_max)
    
    term_g_center = p[7] * g_center
    term_g_min = p[8] * g_min
    term_g_max = p[9] * g_max
    
    return term_bias + term_x + term_y + term_z + term_x2 + term_y2 + term_z2 + term_g_center + term_g_min + term_g_max
# EVOLVE-BLOCK-END
