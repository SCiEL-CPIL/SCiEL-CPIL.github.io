import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a 5th-order polynomial in strain with linear temperature dependence
    for each coefficient, enforcing zero stress at zero strain.
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Unpack parameters
    # c0, c1: Constant term of P (multiplied by eps) -> Linear in eps
    # c2, c3: Eps term of P -> Quadratic in eps
    # c4, c5: Eps^2 term of P -> Cubic in eps
    # c6, c7: Eps^3 term of P -> Quartic in eps
    # c8, c9: Eps^4 term of P -> Quintic in eps
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Calculate powers of strain
    e1 = eps
    e2 = e1 * e1
    e3 = e2 * e1
    e4 = e2 * e2
    e5 = e4 * e1
    
    # Polynomial P(eps, T)
    # P = (c0 + c1*T) + (c2 + c3*T)*eps + (c4 + c5*T)*eps^2 + (c6 + c7*T)*eps^3 + (c8 + c9*T)*eps^4
    p0 = c0 + c1 * T
    p1 = c2 + c3 * T
    p2 = c4 + c5 * T
    p3 = c6 + c7 * T
    p4 = c8 + c9 * T
    
    P = p0 + p1 * e1 + p2 * e2 + p3 * e3 + p4 * e4
    
    # Stress = eps * P
    stress = e1 * P
    
    return stress
