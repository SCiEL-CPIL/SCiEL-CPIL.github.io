"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a combination of polynomial and smooth nonlinear basis functions
    to capture the elastic-plastic transition, work hardening, and
    temperature dependence in the stress-strain response.
    
    The model combines:
    - Polynomial terms in strain up to degree 4
    - Smooth saturation via tanh for plastic yielding
    - Temperature-dependent modifications to modulus and yield strength
    - Cross terms between strain and temperature
    
    params (10 constants):
      p0: Intercept
      p1: Linear strain coeff (elastic modulus approx)
      p2: Quadratic strain coeff
      p3: Cubic strain coeff
      p4: Quartic strain coeff
      p5: Temperature linear coeff
      p6: Temperature-strain interaction
      p7: Saturation amplitude (yield strength approx)
      p8: Saturation rate (controls elastic-plastic transition sharpness)
      p9: Temperature-modulated saturation rate
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Polynomial basis in strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    
    # Temperature basis
    T2 = T * T
    
    # Smooth nonlinear saturation term: captures the elastic-to-plastic transition
    # The argument is scaled by (p8 + p9*T) to allow temperature-dependent transition
    # p7 controls the amplitude (yield strength)
    sat_arg = (p8 + p9 * T) * eps
    # Use tanh for smooth saturation: tanh(x) -> 1 for large x, ~x for small x
    # This gives a smooth approximation of a yield plateau
    sat = np.tanh(sat_arg)
    
    # Construct the full model
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps3 + p4 * eps4
    sigma += p5 * T + p6 * T * eps
    sigma += p7 * sat
    
    return sigma
# EVOLVE-BLOCK-END
