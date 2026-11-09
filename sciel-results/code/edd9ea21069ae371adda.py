"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial model with temperature-dependent coefficients for the
    lower-order terms to capture elastic-plastic transition and hardening.
    """
    s = strain
    t = temp
    
    # Precompute powers of strain
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    
    # Coefficients with linear temperature dependence for terms 0-3
    # c0(T) = p0 + p1*T
    # c1(T) = p2 + p3*T
    # c2(T) = p3 + p5*T (Wait, indexing)
    
    # Let's map params carefully
    # p0, p1: constant term (base, temp coeff)
    # p2, p3: linear strain term (base, temp coeff)
    # p4, p5: quadratic strain term (base, temp coeff)
    # p6, p7: cubic strain term (base, temp coeff)
    # p8: quartic strain term (base only)
    # p9: quintic strain term (base only)
    
    c0 = params[0] + params[1] * t
    c1 = params[2] + params[3] * t
    c2 = params[4] + params[5] * t
    c3 = params[6] + params[7] * t
    c4 = params[8]
    c5 = params[9]
    
    return c0 + c1 * s + c2 * s2 + c3 * s3 + c4 * s4 + c5 * s5
# EVOLVE-BLOCK-END
