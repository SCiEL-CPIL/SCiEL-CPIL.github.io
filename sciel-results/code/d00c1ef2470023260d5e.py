"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Fourier basis with fundamental frequency pi (period 2) for domain [-1, 1].
    This basis is orthogonal over the domain, ensuring numerical stability for BFGS.
    We include all first-order interaction terms to better capture non-separable functions.
    
    Basis functions:
    p0: 1
    p1: cos(pi*x)
    p2: sin(pi*x)
    p3: cos(pi*y)
    p4: sin(pi*y)
    p5: cos(pi*x)*cos(pi*y)
    p6: sin(pi*x)*sin(pi*y)
    p7: cos(pi*x)*sin(pi*y)
    p8: sin(pi*x)*cos(pi*y)
    p9: cos(2*pi*x)  (Second harmonic for x, to capture finer details)
    """
    # Ensure x and y are 1D arrays of shape (N,)
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    # Fundamental frequency: pi (period 2, matches domain width)
    c1x = np.cos(np.pi * x)
    s1x = np.sin(np.pi * x)
    
    c1y = np.cos(np.pi * y)
    s1y = np.sin(np.pi * y)
    
    # Second harmonic: 2*pi (period 1)
    c2x = np.cos(2.0 * np.pi * x)
    
    # Construct prediction
    pred = (
        p0
        + p1 * c1x
        + p2 * s1x
        + p3 * c1y
        + p4 * s1y
        + p5 * c1x * c1y
        + p6 * s1x * s1y
        + p7 * c1x * s1y
        + p8 * s1x * c1y
        + p9 * c2x
    )
    
    return pred
# EVOLVE-BLOCK-END
