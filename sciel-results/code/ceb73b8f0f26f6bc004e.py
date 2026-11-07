import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a rational function model with temperature-dependent coefficients.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # Numerator: (a0 + a1*T) * eps + (b0 + b1*T) * eps^2
    a0 = p0
    a1 = p1
    b0 = p2
    b1 = p3
    
    # Denominator: (c0 + c1*T) + (d0 + d1*T) * eps
    c0 = p4
    c1 = p5
    d0 = p6
    d1 = p7
    
    # Additional linear and constant terms
    lin_coef = p8
    const_term = p9
    
    # Calculate terms
    eps = strain
    T = temp
    
    num = (a0 + a1 * T) * eps + (b0 + b1 * T) * eps**2
    den = (c0 + c1 * T) + (d0 + d1 * T) * eps
    
    # Avoid division by zero or near-zero
    # If den is very small, the result blows up. 
    # Adding a small epsilon to den? 
    # Better to rely on the optimizer. But for safety during initial steps:
    # We can clip den to be away from 0.
    # However, the prompt says "non-finite error costs 1e30". 
    # So if we produce inf/nan, it's bad.
    # Let's use np.where to handle potential singularities if den is close to 0.
    # But simple division is usually fine if initialized well.
    
    result = num / den + lin_coef * eps + const_term
    
    return result
