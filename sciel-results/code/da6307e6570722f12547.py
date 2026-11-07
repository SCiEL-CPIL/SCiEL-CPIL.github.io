import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a rational function model with temperature-dependent coefficients
    to capture elastic-plastic transition, strain hardening, and saturation.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature factor: 1 at low temp, approaches 0 at high temp
    # temp in [0.067, 1], so (1-temp) in [0, 0.933]
    t = 1.0 - temp
    
    # Temperature-dependent coefficients
    a = p0 + p1 * t       # linear (elastic) coefficient
    b = p2 + p3 * t       # quadratic (hardening) coefficient
    c = p4 + p5 * t       # denominator linear coefficient (controls saturation)
    d = p6 + p7 * t       # additional linear term
    e = p8 + p9 * t       # constant offset
    
    # Rational core: (a*eps + b*eps^2) / (1 + c*eps)
    # This gives linear behavior at small eps, saturation at large eps
    eps = strain
    denom = 1.0 + c * eps
    # Guard against zero denominator
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    
    rational = (a * eps + b * eps * eps) / denom
    
    # Full model
    stress = rational + d * eps + e
    
    return stress
