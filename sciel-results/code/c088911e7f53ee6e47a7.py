"""Symbolic regression: closed-form expression with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A flexible closed-form model using a sum of sinusoidal and polynomial terms.
    
    The model is:
      f(x) = p0 + p1*x + p2*x^2 + p3*x^3 
             + p4*sin(a1*x) + p5*cos(b1*x)
             + p6*sin(a2*x) + p7*cos(b2*x)
             + p8*sin(a3*x) + p9*cos(b3*x)
    
    where a1, b1, a2, b2, a3, b3 are fixed frequencies chosen to capture
    a range of oscillatory behaviors. Since we only have 10 params, we use
    them as amplitudes for a fixed set of basis functions.
    
    Actually, let's use a more standard approach: a sum of basis functions
    with fixed centers and widths, or a Fourier-like series.
    
    Let's try: 
      f(x) = sum_{k=0}^{9} p_k * phi_k(x)
    
    where phi_k are fixed basis functions. Good choices:
    - Polynomials: 1, x, x^2, x^3, x^4
    - Sinusoids: sin(x), cos(x), sin(2x), cos(2x), sin(3x)
    
    This gives us 10 basis functions total.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=float).ravel()
    
    # Build basis matrix
    # Basis functions:
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: sin(2x)
    # 8: cos(2x)
    # 9: sin(3x)
    
    basis = np.empty((n, 10), dtype=float)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x * x
    basis[:, 3] = x * x * x
    basis[:, 4] = x * x * x * x
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.sin(2.0 * x)
    basis[:, 8] = np.cos(2.0 * x)
    basis[:, 9] = np.sin(3.0 * x)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
