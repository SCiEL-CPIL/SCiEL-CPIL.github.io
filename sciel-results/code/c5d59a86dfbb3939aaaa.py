import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a rational-polynomial form.
    
    The model captures:
    1. Elastic region with temperature-dependent modulus.
    2. Yield transition via the rational term saturation.
    3. Plastic hardening via polynomial terms in strain.
    4. Temperature effects on hardening and baseline stress.
    """
    eps = strain
    T = temp
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic term: Modulus p0, softened by temperature (1 - p2*T), 
    # saturating as strain increases via (1 + p1*eps) in denominator.
    # This creates a "knee" at strain ~ 1/p1.
    elastic_term = (p0 * eps) / (1.0 + p1 * eps)
    temp_modulator = (1.0 - p2 * T)
    elastic_contribution = elastic_term * temp_modulator
    
    # Plastic/Hardening terms
    # Quadratic and cubic strain terms to model hardening curvature
    hardening = p3 * (eps * eps) + p6 * (eps * eps * eps)
    
    # Temperature interactions with hardening
    temp_hardening = p5 * (T * eps) + p7 * (T * eps * eps)
    
    # Pure temperature terms
    temp_baseline = p4 * T + p8 * (T * T)
    
    # Constant offset
    offset = p9
    
    return elastic_contribution + hardening + temp_hardening + temp_baseline + offset
