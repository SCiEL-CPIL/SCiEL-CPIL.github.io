"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Bivariate polynomial (degree 3 in strain, degree 2 in temp)
    with 10 fitted coefficients. This provides a smooth, differentiable
    surface that captures non-linear interactions between strain and
    temperature without structural bias from physical constitutive laws.
    
    stress = sum_{i=0}^{3} sum_{j=0}^{2} p_{i,j} * strain^i * temp^j
    
    Params (10 total):
    0: p00 - constant term
    1: p10 - linear in strain
    2: p20 - quadratic in strain
    3: p30 - cubic in strain
    4: p01 - linear in temp
    5: p11 - strain * temp
    6: p21 - strain^2 * temp
    7: p31 - strain^3 * temp
    8: p02 - quadratic in temp
    9: p12 - strain * temp^2
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    p00, p10, p20, p30 = params[0], params[1], params[2], params[3]
    p01, p11, p21, p31 = params[4], params[5], params[6], params[7]
    p02, p12 = params[8], params[9]
    
    # Build polynomial terms
    s = strain
    t = temp
    
    # Degree 0 in strain
    term0 = p00 + p01 * t + p02 * t * t
    
    # Degree 1 in strain
    term1 = (p10 + p11 * t + p12 * t * t) * s
    
    # Degree 2 in strain
    term2 = (p20 + p21 * t) * s * s
    
    # Degree 3 in strain
    term3 = (p30 + p31 * t) * s * s * s
    
    result = term0 + term1 + term2 + term3
    
    return result
# EVOLVE-BLOCK-END
