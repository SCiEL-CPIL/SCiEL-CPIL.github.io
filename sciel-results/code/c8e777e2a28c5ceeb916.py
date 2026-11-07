"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a low-order 2D Fourier Series basis.
    
    The model is a linear combination of sinusoidal terms:
    z = p[0]
      + p[1]*cos(pi*x) + p[2]*cos(pi*y)
      + p[3]*sin(pi*x) + p[4]*sin(pi*y)
      + p[5]*cos(pi*x)*cos(pi*y) + p[6]*sin(pi*x)*sin(pi*y)
      + p[7]*cos(2*pi*x) + p[8]*cos(2*pi*y)
      + p[9]*sin(2*pi*x)
    
    This approach is distinct from polynomial regression and leverages
    the bounded domain [-1, 1] effectively with orthogonal-like basis functions.
    """
    p = params.astype(float)
    
    # Precompute trigonometric terms
    pi = np.pi
    
    cos_pi_x = np.cos(pi * x)
    cos_pi_y = np.cos(pi * y)
    sin_pi_x = np.sin(pi * x)
    sin_pi_y = np.sin(pi * y)
    
    cos_2pi_x = np.cos(2 * pi * x)
    cos_2pi_y = np.cos(2 * pi * y)
    sin_2pi_x = np.sin(2 * pi * x)
    
    # Construct the linear combination
    # Note: We use a specific subset of 10 terms to span the parameter space
    # p[0]: 1 (DC term)
    # p[1]: cos(pi*x)
    # p[2]: cos(pi*y)
    # p[3]: sin(pi*x)
    # p[4]: sin(pi*y)
    # p[5]: cos(pi*x)*cos(pi*y)
    # p[6]: sin(pi*x)*sin(pi*y)
    # p[7]: cos(2*pi*x)
    # p[8]: cos(2*pi*y)
    # p[9]: sin(2*pi*x)
    
    result = (p[0] 
              + p[1] * cos_pi_x 
              + p[2] * cos_pi_y 
              + p[3] * sin_pi_x 
              + p[4] * sin_pi_y 
              + p[5] * (cos_pi_x * cos_pi_y) 
              + p[6] * (sin_pi_x * sin_pi_y) 
              + p[7] * cos_2pi_x 
              + p[8] * cos_2pi_y 
              + p[9] * sin_2pi_x)
              
    return result
# EVOLVE-BLOCK-END
