"""Program using a mixed global basis with polynomial, trigonometric, and smooth nonlinear terms."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a well-conditioned global basis of 10 functions:
    - Polynomial terms (degree 0-3)
    - Trigonometric terms (sin, cos)
    - Smooth nonlinear terms (tanh, sigmoid-like)
    
    This provides BFGS with a better-conditioned landscape than RBF kernels
    while maintaining expressiveness for smooth target functions.
    """
    # Scale input to [-1, 1] for numerical stability
    u = x / 3.0
    
    # Build the 10 basis functions
    # 1-4: Polynomial terms
    b0 = np.ones_like(u)
    b1 = u
    b2 = u * u
    b3 = u * u * u
    
    # 5-6: Trigonometric terms
    b4 = np.sin(u)
    b5 = np.cos(u)
    
    # 7-8: Higher frequency trig
    b6 = np.sin(2 * u)
    b7 = np.cos(2 * u)
    
    # 9-10: Smooth nonlinear terms
    b8 = np.tanh(u)
    b9 = np.tanh(2 * u)
    
    # Stack into design matrix (n, 10)
    design_matrix = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination: y = design_matrix @ params
    y = design_matrix @ params
    
    return y
