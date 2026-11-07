"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable Fourier series model:
        A1 * sin(f1*x) * sin(f2*y)
      + A2 * sin(f3*x) * sin(f4*y)
      + e  * x * y
      + f  (constant)
    
    params layout (10 elements):
      [0] A1  - amplitude of first product term
      [1] f1  - frequency of sin in x for first term
      [2] f2  - frequency of sin in y for first term
      [3] A2  - amplitude of second product term
      [4] f3  - frequency of sin in x for second term
      [5] f4  - frequency of sin in y for second term
      [6] e   - coefficient of x*y interaction
      [7] f   - constant offset
      [8], [9] - unused (reserved for future expansion)
    """
    p = params
    A1 = p[0]
    f1 = p[1]
    f2 = p[2]
    A2 = p[3]
    f3 = p[4]
    f4 = p[5]
    e  = p[6]
    f  = p[7]
    
    # Term 1: A1 * sin(f1*x) * sin(f2*y)
    term1 = A1 * np.sin(f1 * x) * np.sin(f2 * y)
    
    # Term 2: A2 * sin(f3*x) * sin(f4*y)
    term2 = A2 * np.sin(f3 * x) * np.sin(f4 * y)
    
    # Interaction and constant
    pred = term1 + term2 + e * x * y + f
    
    return pred
# EVOLVE-BLOCK-END
