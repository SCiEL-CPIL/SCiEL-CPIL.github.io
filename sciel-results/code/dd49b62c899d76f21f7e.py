"""Symbolic regression with rational function approximation using polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation: y = P(x) / (1 + Q(x)^2)
    
    params (length 10):
    - params[0:5]: coefficients for numerator polynomial P(x) of degree 4
    - params[5:10]: coefficients for denominator polynomial Q(x) of degree 4
    
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4) / (1 + (q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4)^2)
    
    The denominator is always >= 1, ensuring stability for BFGS optimization.
    This form can capture rational-like behaviors, hyperbolic trends, and other
    non-polynomial functions that linear Chebyshev bases cannot fit efficiently.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Scale x from [-2, 2] to [-1, 1] for better conditioning
    t = x / 2.0
    
    # Compute powers of t up to degree 4
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    # Numerator polynomial P(t) of degree 4
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    numerator = p0 + p1 * t + p2 * t2 + p3 * t3 + p4 * t4
    
    # Denominator polynomial Q(t) of degree 4
    q0, q1, q2, q3, q4 = params[5], params[6], params[7], params[8], params[9]
    q_poly = q0 + q1 * t + q2 * t2 + q3 * t3 + q4 * t4
    
    # Denominator: 1 + Q(t)^2, always >= 1
    denominator = 1.0 + q_poly * q_poly
    
    # Compute result
    result = numerator / denominator
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
