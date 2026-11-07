"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function architecture: P3(x, y) / (1 + P1(x, y))
    
    Uses 10 parameters to define:
    - Numerator: cubic polynomial in x and y (6 terms)
    - Denominator: 1 + linear polynomial in x and y (3 terms)
    
    This non-linear parametric structure can represent functions with poles,
    asymptotic behavior, or complex rational dependencies.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # safety margin before 120s limit

    # Extract parameters
    # Numerator coefficients (6): for 1, x, y, x^2, xy, y^2
    p0, p1, p2, p3, p4, p5 = params[0], params[1], params[2], params[3], params[4], params[5]
    # Denominator coefficients (3): for x, y, xy (constant is 1)
    p6, p7, p8 = params[6], params[7], params[8]
    # We have 2 more parameters; use them for higher-order numerator terms: x^3, y^3
    p9, p10 = params[9], params[10] if len(params) > 10 else 0.0

    # Compute numerator: cubic polynomial
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    
    numerator = (p0 + 
                 p1 * x + 
                 p2 * y + 
                 p3 * x2 + 
                 p4 * xy + 
                 p5 * y2 +
                 p9 * x3 +
                 p10 * y3)

    # Compute denominator: 1 + linear terms
    denominator = 1.0 + p6 * x + p7 * y + p8 * xy

    # Ensure denominator doesn't cause numerical issues
    # Add a small epsilon to avoid division by zero
    eps = 1e-10
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + eps), denominator)
    
    pred = numerator / safe_denominator

    if time.time() > deadline:
        pass

    return pred
# EVOLVE-BLOCK-END
