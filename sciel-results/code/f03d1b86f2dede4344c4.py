"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a polynomial basis expansion to capture non-linear stress-strain
    behavior and temperature dependence. This approach is numerically stable
    for BFGS optimization compared to deep exponential models, while being
    more flexible than a simple linear model.
    
    Model:
    sigma = p0 + p1*eps + p2*eps^2 + p3*eps^3 + p4*eps^4
            + p5*temp + p6*temp*eps + p7*temp*eps^2 + p8*temp*eps^3
            + p9*temp^2
    
    params (10 constants):
      p0: Intercept (offset)
      p1: Linear strain coeff (Elastic modulus approx)
      p2: Quadratic strain coeff (Initial hardening/curvature)
      p3: Cubic strain coeff (Hardening shape)
      p4: Quartic strain coeff (Saturation effect)
      p5: Linear temp coeff (Yield drop)
      p6: Temp-Strain interaction (Modulus temp dep)
      p7: Temp-Strain^2 interaction
      p8: Temp-Strain^3 interaction
      p9: Temp^2 coeff
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute powers of strain and temp for efficiency and clarity
    eps = strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    
    T = temp
    T2 = T * T
    
    # Construct the polynomial sum
    # Strain terms
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps3 + p4 * eps4
    
    # Temperature terms and interactions
    sigma += p5 * T
    sigma += p6 * T * eps
    sigma += p7 * T * eps2
    sigma += p8 * T * eps3
    sigma += p9 * T2
    
    return sigma
# EVOLVE-BLOCK-END
