"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses dual-tanh model: separate yield transition and hardening saturation.
    """
    e = strain
    t = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Positive strain only for plastic response
    e_pos = np.maximum(e, 0.0)
    
    # Dual tanh: yield transition (sharp) + hardening saturation (gradual)
    # p1: elastic modulus
    # p2: yield strain scale (inverse) - controls sharpness of yield
    # p3: yield stress magnitude
    # p4: hardening strain scale (inverse) - controls saturation rate
    # p5: hardening stress magnitude
    elastic = p1 * e
    yield_term = p3 * np.tanh(p2 * e_pos)
    hardening_term = p5 * np.tanh(p4 * e_pos)
    
    # Temperature corrections
    # p6: thermal stress offset
    # p7: temp-modulated yield
    # p8: temp-modulated hardening
    temp_offset = p6 * (t - 0.5)
    temp_yield = p7 * (t - 0.5) * np.tanh(p2 * e_pos)
    temp_hard = p8 * (t - 0.5) * np.tanh(p4 * e_pos)
    
    # Residual nonlinearity
    # p9: quadratic correction for strain
    quad = p9 * e**2
    
    return (p0 + elastic + yield_term + hardening_term + 
            temp_offset + temp_yield + temp_hard + quad)
# EVOLVE-BLOCK-END
