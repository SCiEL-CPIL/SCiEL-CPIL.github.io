"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function with 10 parameters to approximate hidden function f(x).
    
    The evaluator fits params (length 10) via BFGS minimizing training MSE.
    We design a flexible basis that can represent many common function families:
    - Polynomials up to degree 4
    - Trigonometric functions
    - Gaussian
    - Rational / sigmoid-like
    
    The key insight: BFGS fits the 10 params. We want a basis that is:
    1. Linear in params (so BFGS works well - actually BFGS works on any smooth objective)
    2. Rich enough to approximate many functions
    3. Well-conditioned so BFGS converges from all-ones start
    
    Since BFGS is used (not least squares), the objective is MSE which is quadratic
    in params if the model is linear in params. So a linear-in-params model is ideal
    for BFGS - it will converge to the global optimum quickly.
    
    Let's use a linear combination of 10 basis functions. This makes the problem
    a linear least squares problem, which BFGS will solve efficiently.
    
    Basis functions chosen for [-2, 2]:
    b0 = 1
    b1 = x
    b2 = x^2
    b3 = x^3
    b4 = x^4
    b5 = sin(x)
    b6 = cos(x)
    b7 = exp(-x^2)
    b8 = tanh(x)
    b9 = x * exp(-x^2)  (odd Gaussian-like)
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Build basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-x ** 2)
    b8 = np.tanh(x)
    b9 = x * np.exp(-x ** 2)
    
    # Linear combination
    result = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
              p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
    
    return result
# EVOLVE-BLOCK-END
