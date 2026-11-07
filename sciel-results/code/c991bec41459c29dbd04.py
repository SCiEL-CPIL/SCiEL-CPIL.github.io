"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 5 fixed centers and 5 learnable amplitudes.
    The 10 params are split: first 5 are amplitudes, last 5 are widths (log-scale).
    Centers are fixed at equally spaced points in [-2, 2].
    
    We use a hybrid approach:
    - Use a mixture of smooth basis functions (sin, cos, polynomials) that can
      represent a wide variety of common functions.
    - The basis is designed to be well-conditioned for BFGS.
    
    Actually, let's use a more flexible approach: a sum of sinusoidal and polynomial
    terms that can capture periodic, polynomial, and mixed behaviors.
    
    Design:
    params[0..2]: coefficients for polynomial terms (1, x, x^2)
    params[3..5]: amplitudes for sin(x), cos(x), sin(2x)
    params[6..7]: amplitudes for cos(2x), sin(3x)
    params[8..9]: amplitudes for cos(3x), x^3
    """
    p = params
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    # Polynomial basis
    f = (p[0] * np.ones_like(x) + 
         p[1] * x + 
         p[2] * x**2 +
         p[9] * x**3)
    
    # Sinusoidal basis
    f = f + (p[3] * np.sin(x) + 
             p[4] * np.cos(x) + 
             p[5] * np.sin(2*x) +
             p[6] * np.cos(2*x) +
             p[7] * np.sin(3*x) +
             p[8] * np.cos(3*x))
    
    return f
# EVOLVE-BLOCK-END
