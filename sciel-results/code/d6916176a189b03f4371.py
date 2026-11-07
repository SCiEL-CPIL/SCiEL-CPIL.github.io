"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature using a bivariate polynomial model."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers of strain and temp
    s = strain
    t = temp
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    t2 = t * t
    
    # Bivariate polynomial basis
    # p0: constant
    # p1: strain
    # p2: strain^2
    # p3: strain^3
    # p4: strain^4
    # p5: temp
    # p6: strain * temp
    # p7: temp^2
    # p8: strain^2 * temp
    # p9: strain * temp^2
    
    stress = (p0 + 
              p1 * s + 
              p2 * s2 + 
              p3 * s3 + 
              p4 * s4 + 
              p5 * t + 
              p6 * (s * t) + 
              p7 * t2 + 
              p8 * (s2 * t) + 
              p9 * (s * t2))
              
    return stress
# EVOLVE-BLOCK-END
