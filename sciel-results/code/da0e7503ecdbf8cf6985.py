"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
    sigma = (p0 + p1*T)*eps 
          + (p2 + p3*T)*eps^2 
          + (p4 + p5*T)*tanh(p6*eps + p7) * (1 + p8*eps + p9*T*eps)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear elastic term
    term1 = (p0 + p1 * temp) * strain
    
    # Quadratic strain term
    term2 = (p2 + p3 * temp) * strain**2
    
    # Non-linear saturation/hardening term
    # tanh ensures smoothness and boundedness
    # p7 allows shifting the yield point
    # (1 + p8*eps + p9*T*eps) allows for strain hardening and temp-strain interaction
    term3 = (p4 + p5 * temp) * np.tanh(p6 * strain + p7) * (1.0 + p8 * strain + p9 * temp * strain)
    
    return term1 + term2 + term3
# EVOLVE-BLOCK-END
