"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a 4th-order polynomial in strain with linear temperature dependence
    for each coefficient. This avoids the potential instabilities of rational
    functions (poles) and provides a flexible global fit.
    """
    s = strain
    t = temp
    
    # Extract parameters
    # p_i are base coefficients, q_i are temperature coefficients
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    q0 = params[5]
    q1 = params[6]
    q2 = params[7]
    q3 = params[8]
    q4 = params[9]
    
    # Compute effective coefficients
    c0 = p0 + q0 * t
    c1 = p1 + q1 * t
    c2 = p2 + q2 * t
    c3 = p3 + q3 * t
    c4 = p4 + q4 * t
    
    # Compute powers of strain
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    # Polynomial evaluation
    stress = c0 + c1 * s + c2 * s2 + c3 * s3 + c4 * s4
    
    return stress
# EVOLVE-BLOCK-END
