import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a high-order odd polynomial expansion with temperature-dependent coefficients.
    This allows for complex curvature (yielding, hardening) while maintaining smoothness
    and odd symmetry (compression/tension symmetry).
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers of strain for efficiency
    eps2 = eps * eps
    eps3 = eps * eps2
    eps5 = eps3 * eps2
    eps7 = eps5 * eps2
    eps9 = eps7 * eps2
    
    # Temperature-dependent coefficients
    c0 = p0 + p1 * T
    c1 = p2 + p3 * T
    c2 = p4 + p5 * T
    c3 = p6 + p7 * T
    c4 = p8 + p9 * T
    
    # Sum of odd polynomial terms
    stress = c0 * eps + c1 * eps3 + c2 * eps5 + c3 * eps7 + c4 * eps9
    
    return stress
