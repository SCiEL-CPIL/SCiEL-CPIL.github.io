"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature (Polynomial Model)."""
    # params: length 10
    # p0: intercept
    # p1: strain coeff
    # p2: strain^2 coeff
    # p3: strain^3 coeff
    # p4: strain^4 coeff
    # p5: temp coeff
    # p6: strain*temp coeff
    # p7: strain^2*temp coeff
    # p8: temp^2 coeff
    # p9: strain^3*temp coeff

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x = strain
    t = temp

    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    t2 = t * t

    xt = x * t
    x2t = x2 * t
    x3t = x3 * t

    stress = (p0 + 
              p1 * x + 
              p2 * x2 + 
              p3 * x3 + 
              p4 * x4 + 
              p5 * t + 
              p6 * xt + 
              p7 * x2t + 
              p8 * t2 + 
              p9 * x3t)

    return stress
# EVOLVE-BLOCK-END
