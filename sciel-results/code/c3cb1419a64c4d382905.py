"""Rewritten symbolic regression using a basis of polynomials, logarithms, exponentials, and local RBFs."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a diverse global and local basis.
    
    Basis (10 terms):
    b0 = 1
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = log(1 + x)
    b5 = exp(-x)
    b6 = exp(-0.5 * x)
    b7 = exp(-2 * x)
    b8 = exp(-(x - 2.0)**2)
    b9 = exp(-(x - 4.0)**2)
    
    Rationale:
    - Polynomials (1, x, x^2, x^3) capture general trends.
    - Logarithm (log(1+x)) captures slow growth.
    - Exponentials (exp(-x), exp(-0.5x), exp(-2x)) capture various decay rates, 
      which were found to be crucial in previous improvements.
    - Two RBFs centered at 2.0 and 4.0 with width 1.0 provide local flexibility 
      to fit features that global functions miss, without the over-parameterization 
      of 5 RBFs which may harm conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = params
    
    # Global Basis
    b0 = 1.0
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = np.log1p(x)  # log(1 + x)
    
    # Exponential Basis
    b5 = np.exp(-x)
    b6 = np.exp(-0.5 * x)
    b7 = np.exp(-2.0 * x)
    
    # Local Radial Basis Functions (Gaussian kernels)
    # Centers chosen to cover the middle and upper part of the domain [0.1, 6]
    # Width sigma = 1.0 (implicit in exponent - (x-c)^2)
    b8 = np.exp(-(x - 2.0) ** 2)
    b9 = np.exp(-(x - 4.0) ** 2)
    
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
