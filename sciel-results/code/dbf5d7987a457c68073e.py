"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stable rational-like approximation using a squared denominator to ensure
    positivity and avoid division by zero issues.
    
    y = p0 + (p1 + p2*x + p3*x^2 + p4*x^3 + p5*x^4 + p6*x^5) / (1 + (q1*x + q2*x^2 + q3*x^3)^2)
    
    params: 
    [0] p0: constant offset
    [1] p1: numerator x^0 coeff
    [2] p2: numerator x^1 coeff
    [3] p3: numerator x^2 coeff
    [4] p4: numerator x^3 coeff
    [5] p5: numerator x^4 coeff
    [6] p6: numerator x^5 coeff
    [7] q1: denominator inner x^1 coeff
    [8] q2: denominator inner x^2 coeff
    [9] q3: denominator inner x^3 coeff
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
    
    # Compute numerator: p1 + p2*x + p3*x^2 + p4*x^3 + p5*x^4 + p6*x^5
    numerator = p1 + p2 * x + p3 * x2 + p4 * x3 + p5 * x4 + p6 * x5
    
    # Compute inner denominator term: q1*x + q2*x^2 + q3*x^3
    inner_denom = q1 * x + q2 * x2 + q3 * x3
    
    # Compute denominator: 1 + inner_denom^2
    # This ensures denominator >= 1, preventing division by zero and keeping gradients stable
    denominator = 1.0 + inner_denom * inner_denom
    
    # Compute rational part
    rational_part = numerator / denominator
    
    # Add constant offset
    result = p0 + rational_part
    
    # Ensure result is finite (though it should be with this form)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
