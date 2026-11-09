"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a quadratic response surface on centered inputs.
    This structure captures curvature (saturation, peaks) without the initialization
    difficulties of Gaussian/Haldane terms, as centers are fixed.
    """
    # Hardcoded centers to aid optimization from params=ones
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    p = params
    
    # Quadratic terms
    b2 = b_c * b_c
    s2 = s_c * s_c
    t2 = t_c * t_c
    p2 = p_c * p_c
    
    # Interaction terms
    bs = b_c * s_c
    bt = b_c * t_c
    bp = b_c * p_c
    st = s_c * t_c
    sp = s_c * p_c
    tp = t_c * p_c
    
    # Linear terms
    l_b = b_c
    l_s = s_c
    l_t = t_c
    l_p = p_c
    
    # Construct prediction
    pred = (
        p[0] * l_b +
        p[1] * l_s +
        p[2] * l_t +
        p[3] * l_p +
        p[4] * b2 +
        p[5] * s2 +
        p[6] * t2 +
        p[7] * p2 +
        p[8] * bs +
        p[9] * st
    )
    
    return pred
# EVOLVE-BLOCK-END
