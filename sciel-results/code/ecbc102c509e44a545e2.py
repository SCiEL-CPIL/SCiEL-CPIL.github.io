"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation with guaranteed positive denominator:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5 + p6*x^6) / (1 + (p7*x + p8*x^2 + p9*x^3)^2)
    
    The denominator is always >= 1, ensuring numerical stability for BFGS.
    All 10 parameters are used: 7 for numerator, 3 for the squared denominator term.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    q1 = params[7]
    q2 = params[8]
    q3 = params[9]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    
    # Compute numerator: degree 6 polynomial
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4 + p5 * x5 + p6 * x6
    
    # Compute denominator: 1 + (q1*x + q2*x^2 + q3*x^3)^2
    # This is always >= 1, so no division by zero or negative values
    inner = q1 * x + q2 * x2 + q3 * x3
    denominator = 1.0 + inner * inner
    
    result = numerator / denominator
    
    # Ensure result is finite (should always be, but safety check)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
