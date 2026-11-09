"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear model with parameterized yield points and
    work-hardening segments to capture the kinked transitions typical
    of aluminum stress-strain curves.
    
    Model:
    sigma = E(T)*eps + Y(T)*hinge(eps, y1) + H(T)*hinge(eps, y2) + H2(T)*hinge(eps, y3) + offset
    where hinge(x, c) = max(0, x - c)
    E(T) = p0 + p1*T
    Y(T) = p2 + p3*T  
    H(T) = p4 + p5*T
    H2(T) = p6 + p7*T
    offset = p8 + p9*T
    
    Parameters (10 total):
    p0, p1: elastic modulus coefficients
    p2, p3: yield strength coefficients
    p4, p5: first hardening slope coefficients
    p6, p7: second hardening slope coefficients
    p8: constant offset
    p9: temperature offset coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T
    Y_T = p2 + p3 * T
    H_T = p4 + p5 * T
    H2_T = p6 + p7 * T
    offset_T = p8 + p9 * T
    
    # Piecewise-linear hinge functions: max(0, eps - c)
    # Using fixed yield points based on typical aluminum behavior
    # y1 ~ 0.002 (initial yield), y2 ~ 0.01 (onset of significant hardening), y3 ~ 0.05 (strain hardening regime)
    y1 = 0.002
    y2 = 0.01
    y3 = 0.05
    
    h1 = np.maximum(0.0, eps - y1)
    h2 = np.maximum(0.0, eps - y2)
    h3 = np.maximum(0.0, eps - y3)
    
    # Core model: elastic + yield + hardening segments + offset
    elastic_term = E_T * eps
    yield_term = Y_T * h1
    hardening_term = H_T * h2
    hardening2_term = H2_T * h3
    
    sigma = elastic_term + yield_term + hardening_term + hardening2_term + offset_T
    
    return sigma
# EVOLVE-BLOCK-END
