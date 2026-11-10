"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P2(x) * Q7(x)
    P2 is a degree-2 polynomial (3 coeffs).
    Q7 is a degree-7 polynomial (7 coeffs).
    Total 10 params.
    Product is a degree-9 polynomial.
    
    This structure allows for a quadratic envelope multiplied by a higher-order polynomial,
    which may better capture functions with varying curvature or oscillatory behavior
    compared to the symmetric P4*Q4 split.
    """
    # P(x) = p0 + p1*x + p2*x^2
    p0, p1, p2 = params[0], params[1], params[2]
    Px = (p2 * x + p1) * x + p0
    
    # Q(x) = q0 + q1*x + ... + q7*x^7
    # Horner's method
    q = params[3:10]
    # q[0] is q0, ..., q[6] is q6, q[7] is q7? 
    # Wait, params[3:10] has 7 elements.
    # Let q0..q6 be the coefficients.
    # Q(x) = q0 + q1*x + ... + q6*x^6 ?
    # No, Q7 needs 8 coeffs (deg 7).
    # I allocated 7 params for Q. So Q is degree 6.
    # Let's adjust: P2 (3 coeffs) * Q6 (7 coeffs) = 10 params. Degree 8.
    # Or P1 (2 coeffs) * Q7 (8 coeffs) = 10 params. Degree 8.
    
    # Let's stick to P2 * Q6.
    # P2: deg 2. Q6: deg 6.
    q0, q1, q2, q3, q4, q5, q6 = params[3:10]
    Qx = q6 * x**5 + q5 * x**4 + q4 * x**3 + q3 * x**2 + q2 * x + q1
    Qx = Qx * x + q0
    
    return Px * Qx
# EVOLVE-BLOCK-END
