"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a mixture of basis functions to recover an unknown function.
    
    The 10 parameters are split into:
    - 5 coefficients for basis functions
    - 5 parameters that shape the basis functions (frequencies, exponents, etc.)
    
    This design allows local feature interpolation while remaining differentiable
    for BFGS optimization.
    """
    # Unpack parameters
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Ensure numeric stability by clipping exponent arguments
    # Basis 1: Gaussian centered at c5 with width related to c6
    g1_arg = -((x - c5) ** 2) / (c6 ** 2 + 1e-8)
    g1_arg = np.clip(g1_arg, -50, 50)
    basis1 = c0 * np.exp(g1_arg)
    
    # Basis 2: Gaussian centered at c7 with width related to c8
    g2_arg = -((x - c7) ** 2) / (c8 ** 2 + 1e-8)
    g2_arg = np.clip(g2_arg, -50, 50)
    basis2 = c1 * np.exp(g2_arg)
    
    # Basis 3: Polynomial component (low degree)
    basis3 = c2 * x + c3 * x ** 2
    
    # Basis 4: Trigonometric component with frequency c9
    basis4 = c4 * np.sin(c9 * x)
    
    # Combine all basis functions
    result = basis1 + basis2 + basis3 + basis4
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
