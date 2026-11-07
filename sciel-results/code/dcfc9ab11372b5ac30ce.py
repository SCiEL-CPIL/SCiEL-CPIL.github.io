import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Polynomial in strain with linear temperature interactions.
    sigma = sum_{i=0}^4 (p_i + p_{i+5} * T) * strain^i
    """
    e = strain
    T = temp
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute powers of strain
    e1 = e
    e2 = e1 * e1
    e3 = e2 * e1
    e4 = e2 * e2
    
    # Compute base terms (no temp)
    t0 = p0
    t1 = p1 * e1
    t2 = p2 * e2
    t3 = p3 * e3
    t4 = p4 * e4
    
    # Compute temp interaction terms
    u0 = p5
    u1 = p6 * e1
    u2 = p7 * e2
    u3 = p8 * e3
    u4 = p9 * e4
    
    # Combine
    return t0 + t1 + t2 + t3 + t4 + T * (u0 + u1 + u2 + u3 + u4)
