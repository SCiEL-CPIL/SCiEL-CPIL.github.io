"""Symbolic regression with 1 input using a well-conditioned diverse basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions (10 total):
    0: T_0(x/3) = 1
    1: T_1(x/3) = x/3
    2: T_2(x/3) = 2(x/3)^2 - 1
    3: T_3(x/3) = 4(x/3)^3 - 3(x/3)
    4: sin(x)
    5: cos(x)
    6: sin(2x)
    7: cos(2x)
    8: tanh(x)
    9: tanh(2x)
    
    Using Chebyshev polynomials scaled to the domain [-3, 3] improves
    numerical conditioning for the BFGS optimizer compared to raw monomials.
    The mix of polynomial, trigonometric, and saturating (tanh) functions
    provides flexibility to fit periodic, polynomial, and step-like features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_3
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # Trigonometric functions
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    
    # Saturating functions (soft-step approximations)
    b8 = np.tanh(x)
    b9 = np.tanh(2.0 * x)
    
    # Stack into a matrix (N, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
