"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial chaos expansion (Taylor series) for stress-strain
    of an aluminium rod.
    
    Uses up to 10 coefficients to construct a smooth, differentiable surface:
    sigma = sum_{i,j} c_{ij} * strain^i * temp^j
    
    with i + j <= 3 (giving 1+2+3+4 = 10 terms for degrees 0,1,2,3)
    
    Terms:
    p[0]:  c00
    p[1]:  c10
    p[2]:  c01
    p[3]:  c20
    p[4]:  c11
    p[5]:  c02
    p[6]:  c30
    p[7]:  c21
    p[8]:  c12
    p[9]:  c03
    """
    s = strain
    t = temp
    
    # Degree 0
    c00 = params[0]
    # Degree 1
    c10 = params[1]
    c01 = params[2]
    # Degree 2
    c20 = params[3]
    c11 = params[4]
    c02 = params[5]
    # Degree 3
    c30 = params[6]
    c21 = params[7]
    c12 = params[8]
    c03 = params[9]
    
    return (c00 
            + c10 * s + c01 * t 
            + c20 * s**2 + c11 * s * t + c02 * t**2
            + c30 * s**3 + c21 * s**2 * t + c12 * s * t**2 + c03 * t**3)
# EVOLVE-BLOCK-END
