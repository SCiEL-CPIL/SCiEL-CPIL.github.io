"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P(x) * Q(x) where P is degree 5 (6 coeffs) and Q is degree 3 (4 coeffs).
    Total 10 parameters. Product is a degree 8 polynomial.
    
    P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    Q(x) = q0 + q1*x + q2*x^2 + q3*x^3
    
    params[0:6] are p0..p5
    params[6:10] are q0..q3
    """
    p0, p1, p2, p3, p4, p5 = params[0:6]
    q0, q1, q2, q3 = params[6:10]
    
    # Compute P(x) using Horner's method
    Px = p5
    Px = Px * x + p4
    Px = Px * x + p3
    Px = Px * x + p2
    Px = Px * x + p1
    Px = Px * x + p0
    
    # Compute Q(x) using Horner's method
    Qx = q3
    Qx = Qx * x + q2
    Qx = Qx * x + q1
    Qx = Qx * x + q0
    
    return Px * Qx
# EVOLVE-BLOCK-END
