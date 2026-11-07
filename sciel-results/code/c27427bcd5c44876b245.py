import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rational function structure.
    
    Structure:
    f(x, y, z) = (p0*x + p1*y + p2*z + p3) / (p4*x + p5*y + p6*z + p7 + 1.0) + p8*x + p9*y
    
    This form allows for:
    1. Rational dependencies (e.g., 1/x, x/y).
    2. Linear dependencies (via the linear tail and the limit of the rational part).
    3. Asymptotic behavior.
    
    The +1.0 in the denominator ensures numerical stability for inputs in [0.5, 2]
    and parameters near 1, preventing division by zero or near-zero values.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    numerator = p0 * x + p1 * y + p2 * z + p3
    denominator = p4 * x + p5 * y + p6 * z + p7 + 1.0
    
    # Safe division
    # The denominator is guaranteed to be > 0 for x,y,z in [0.5, 2] and params near 1.
    # If params drift, the +1.0 helps, but we should be cautious.
    # For robustness against extreme param values during optimization:
    # We can clip the denominator if it gets too small, but standard BFGS
    # on smooth functions usually handles this if the initial point is safe.
    
    rational_part = numerator / denominator
    
    linear_part = p8 * x + p9 * y
    
    return rational_part + linear_part
