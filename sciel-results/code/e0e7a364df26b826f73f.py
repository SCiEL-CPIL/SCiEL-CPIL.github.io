"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative/modulated structure where temperature scales
    the elastic modulus and the yield/hardening magnitude.
    The hardening curve is a rational function to allow flexible
    curvature (concave/convex) and saturation.
    
    Model: sigma = E(T)*eps + Y(T)*H(eps) + C
    E(T) = p0 + p1*T + p2*T^2
    Y(T) = p3 + p4*T
    H(eps) = (p5*eps + p6*eps^2) / (1 + p7*eps + p8*eps^2)
    C = p9
    
    params (10 constants):
      p0, p1, p2: Elastic modulus coefficients
      p3, p4: Yield/Hardening scale coefficients
      p5, p6: Hardening numerator coefficients
      p7, p8: Hardening denominator coefficients
      p9: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    T2 = T * T
    eps2 = eps * eps
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * T + p2 * T2
    
    # Temperature-dependent hardening scale
    Y_T = p3 + p4 * T
    
    # Rational hardening function
    # Numerator: p5*eps + p6*eps^2
    # Denominator: 1 + p7*eps + p8*eps^2
    # Safety: ensure denominator is not near zero
    denom = 1.0 + p7 * eps + p8 * eps2
    
    # Use a safe division. If denom is close to 0, replace with a small value
    # to avoid NaN/Inf. This might distort the fit slightly if it triggers,
    # but prevents total failure.
    safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    
    H_eps = (p5 * eps + p6 * eps2) / safe_denom
    
    sigma = E_T * eps + Y_T * H_eps + p9
    
    return sigma
# EVOLVE-BLOCK-END
