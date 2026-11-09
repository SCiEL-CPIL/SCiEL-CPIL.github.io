import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a polynomial expansion with temperature interactions.
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic and plastic contributions as polynomial in strain
    # Modulated by temperature
    term_eps = eps
    term_eps2 = eps * eps
    term_eps3 = term_eps2 * eps
    term_eps4 = term_eps2 * term_eps2
    
    # Base stress terms (independent of T)
    stress_base = p0 * term_eps + p1 * term_eps2 + p2 * term_eps3 + p3 * term_eps4 + p4
    
    # Temperature dependent terms
    stress_temp = p5 * T + p6 * T * term_eps + p7 * T * term_eps2 + p8 * T * term_eps3 + p9 * T * term_eps4
    
    return stress_base + stress_temp
