"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Pre-compute a rich basis of functions that can be combined linearly.
# The target function is unknown, but for 1D symbolic regression tasks,
# it's often a smooth function that can be approximated by a series of
# basis functions. We'll use a combination of polynomials, trigonometric
# functions, and exponential terms to span a wide space of possible functions.

# Since we don't know the function, we'll use a very flexible model:
# A linear combination of many basis functions. The parameters will be
# fitted by the evaluator using BFGS.

# We need to be careful about:
# 1. Numerical stability (avoid overflow/underflow)
# 2. Computational efficiency (must finish within 150s)
# 3. The model should be rich enough to approximate common functions

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model: linear combination of basis functions.
    
    params[0:10] are used as coefficients for the basis functions.
    We define 10 basis functions that span a wide range of possible behaviors.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define basis functions
    # We use a diverse set of smooth functions to span a wider space.
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic
    b2 = x ** 2
    
    # 4. Cubic
    b3 = x ** 3
    
    # 5. Quartic
    b4 = x ** 4
    
    # 6. Sine
    b5 = np.sin(x)
    
    # 7. Cosine
    b6 = np.cos(x)
    
    # 8. Hyperbolic Tangent
    b7 = np.tanh(x)
    
    # 9. Exponential
    b8 = np.exp(x)
    
    # 10. Negative Exponential
    b9 = np.exp(-x)
    
    # Combine with parameters
    result = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    return result

# EVOLVE-BLOCK-END
