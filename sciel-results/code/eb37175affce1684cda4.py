"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a 4th-order polynomial in centered strain with linear temperature dependence.
    """
    # Center inputs to improve numerical conditioning for polynomial fitting
    # Strain range approx [-0.002, 0.96], center at 0.5
    # Temp range approx [0.06, 1.0], center at 0.5
    eps_c = strain - 0.5
    t_c = temp - 0.5
    
    # Compute powers of centered strain
    eps_c2 = eps_c * eps_c
    eps_c3 = eps_c2 * eps_c
    eps_c4 = eps_c2 * eps_c2
    
    # Extract parameters
    # a_i + b_i * T_c
    # i=0: p0 + p1*t_c
    # i=1: p2 + p3*t_c
    # i=2: p4 + p5*t_c
    # i=3: p6 + p7*t_c
    # i=4: p8 + p9*t_c
    
    # Calculate terms
    term0 = (params[0] + params[1] * t_c)
    term1 = (params[2] + params[3] * t_c) * eps_c
    term2 = (params[4] + params[5] * t_c) * eps_c2
    term3 = (params[6] + params[7] * t_c) * eps_c3
    term4 = (params[8] + params[9] * t_c) * eps_c4
    
    return term0 + term1 + term2 + term3 + term4
# EVOLVE-BLOCK-END
