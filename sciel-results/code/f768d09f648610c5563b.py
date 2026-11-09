"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a rational function (ratio of polynomials) with temperature-dependent
    coefficients to better capture non-linear hardening and potential asymptotes
    compared to pure polynomials.
    """
    s = strain
    t = temp
    
    # Precompute powers of strain
    s2 = s * s
    s3 = s2 * s
    
    # Numerator coefficients with linear temperature dependence
    # N(s, T) = p0 + (p1 + p2*T)*s + (p3 + p4*T)*s^2 + (p5 + p6*T)*s^3
    n0 = params[0]
    n1 = params[1] + params[2] * t
    n2 = params[3] + params[4] * t
    n3 = params[5] + params[6] * t
    
    numerator = n0 + n1 * s + n2 * s2 + n3 * s3
    
    # Denominator coefficients with linear temperature dependence
    # D(s, T) = 1 + (p7 + p8*T)*s + p9*s^2
    # Note: constant term fixed to 1 to reduce parameter redundancy and improve stability
    d1 = params[7] + params[8] * t
    d2 = params[9]
    
    denominator = 1.0 + d1 * s + d2 * s2
    
    # Avoid division by zero issues during optimization
    # Add a small epsilon to denominator if it gets too close to 0
    # However, simple division is often fine if the optimizer doesn't hit singularities.
    # For safety, we can use np.where or just rely on the optimizer.
    # Given the constraints, direct division is standard.
    
    return numerator / denominator
# EVOLVE-BLOCK-END
