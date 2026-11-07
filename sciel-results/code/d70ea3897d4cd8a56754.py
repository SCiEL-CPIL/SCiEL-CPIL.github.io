"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a centered polynomial basis.
    Centers t around 40 to improve numerical conditioning for the optimizer.
    Includes linear, quadratic, and cross terms.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Center variables to reduce magnitude and improve optimization stability
    t_c = t - 40.0
    x_c = x
    v_c = v
    
    # Basis functions
    # 1. Constant
    # 2. Linear t
    # 3. Linear x
    # 4. Linear v
    # 5. Quadratic x
    # 6. Quadratic v
    # 7. Cross x*v
    # 8. Cross t*x
    # 9. Cross t*v
    # 10. Cubic t (approximates sinusoidal drive curvature)
    
    result = (c0 
              + c1 * t_c 
              + c2 * x_c 
              + c3 * v_c 
              + c4 * x_c**2 
              + c5 * v_c**2 
              + c6 * x_c * v_c 
              + c7 * t_c * x_c 
              + c8 * t_c * v_c 
              + c9 * t_c**3)
              
    return result
# EVOLVE-BLOCK-END
