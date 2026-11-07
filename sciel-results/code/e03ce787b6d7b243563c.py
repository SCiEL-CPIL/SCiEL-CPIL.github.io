"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial model in strain with temperature-dependent coefficients.
    """
    # Unpack parameters
    c0 = params[0]
    e0, e1, e2 = params[1], params[2], params[3]
    h0, h1 = params[4], params[5]
    k0, k1 = params[6], params[7]
    l0, l1 = params[8], params[9]
    
    # Precompute temperature terms
    t = temp
    t2 = t * t
    
    # Coefficients for each power of strain
    # Linear term coefficient: E(T) = e0 + e1*T + e2*T^2
    coeff_1 = e0 + e1 * t + e2 * t2
    
    # Quadratic term coefficient: H(T) = h0 + h1*T
    coeff_2 = h0 + h1 * t
    
    # Cubic term coefficient: K(T) = k0 + k1*T
    coeff_3 = k0 + k1 * t
    
    # Quartic term coefficient: L(T) = l0 + l1*T
    coeff_4 = l0 + l1 * t
    
    # Compute powers of strain
    s = strain
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    # Combine
    stress = c0 + coeff_1 * s + coeff_2 * s2 + coeff_3 * s3 + coeff_4 * s4
    
    return stress
# EVOLVE-BLOCK-END
