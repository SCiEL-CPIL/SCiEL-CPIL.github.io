"""Symbolic regression with a rich, stable linear basis for BFGS-compatible optimization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a linear combination of 10 basis functions.
    
    The basis functions are chosen to be diverse and stable:
    1. Constant (1)
    2. Linear (x)
    3. Quadratic (x^2)
    4. Cubic (x^3)
    5. Quartic (x^4)
    6. Sine (sin(pi*x))
    7. Cosine (cos(pi*x))
    8. Sine high freq (sin(2*pi*x))
    9. Gaussian (exp(-x^2))
    10. Tanh (tanh(x))
    
    This linear formulation ensures the objective is convex in the parameters,
    allowing BFGS to converge reliably to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure exactly 10 parameters; pad or truncate if needed
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Compute basis functions
    b1 = np.ones_like(x)
    b2 = x
    b3 = x**2
    b4 = x**3
    b5 = x**4
    b6 = np.sin(np.pi * x)
    b7 = np.cos(np.pi * x)
    b8 = np.sin(2.0 * np.pi * x)
    b9 = np.exp(-x**2)
    b10 = np.tanh(x)
    
    # Linear combination
    result = (p[0] * b1 + p[1] * b2 + p[2] * b3 + p[3] * b4 + p[4] * b5 +
              p[5] * b6 + p[6] * b7 + p[7] * b8 + p[8] * b9 + p[9] * b10)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
