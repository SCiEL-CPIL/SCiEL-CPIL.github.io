"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a rich basis of polynomial, trigonometric,
    and other functional forms to recover an unknown function.
    
    The basis is designed to be flexible enough to capture many common function
    shapes while remaining well-conditioned for BFGS optimization starting from
    all-ones.
    
    x is in [0.1, 6]. We use multiple scaled versions to capture different
    frequency behaviors.
    
    Basis (10 parameters):
    p0: constant
    p1: x
    p2: x^2
    p3: x^3
    p4: sin(x)
    p5: cos(x)
    p6: sin(2*x)
    p7: cos(2*x)
    p8: exp(-x)
    p9: 1/x
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite values
    x = np.where(np.isfinite(x), x, 1.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    
    # Trigonometric terms
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    
    # Exponential and rational terms
    b8 = np.exp(-x)
    b9 = 1.0 / np.maximum(x, 1e-6)  # avoid division by zero
    
    # Combine
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
