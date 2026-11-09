"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Fourier basis with distinct integer frequencies for x and y.
    We use 10 parameters total, allocated to specific basis functions.
    
    Basis functions selected to capture global periodicity and smooth interactions:
    p0: constant
    p1: cos(2*pi*x)
    p2: sin(2*pi*x)
    p3: cos(2*pi*y)
    p4: sin(2*pi*y)
    p5: cos(4*pi*x)
    p6: sin(4*pi*x)
    p7: cos(4*pi*y)
    p8: sin(4*pi*y)
    p9: cos(2*pi*x) * cos(2*pi*y)  (interaction term)
    
    Note: The evaluator fits 10 params starting from all ones using BFGS.
    The basis is fixed; only the coefficients are learned.
    """
    # Ensure x and y are 1D arrays of shape (N,)
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    # Frequencies: 1 and 2 (i.e., 2*pi*1 and 2*pi*2)
    c1x = np.cos(2.0 * np.pi * x)
    s1x = np.sin(2.0 * np.pi * x)
    c2x = np.cos(4.0 * np.pi * x)
    s2x = np.sin(4.0 * np.pi * x)
    
    c1y = np.cos(2.0 * np.pi * y)
    s1y = np.sin(2.0 * np.pi * y)
    c2y = np.cos(4.0 * np.pi * y)
    s2y = np.sin(4.0 * np.pi * y)
    
    # Construct prediction
    pred = (
        p0
        + p1 * c1x
        + p2 * s1x
        + p3 * c1y
        + p4 * s1y
        + p5 * c2x
        + p6 * s2x
        + p7 * c2y
        + p8 * s2y
        + p9 * c1x * c1y
    )
    
    return pred
# EVOLVE-BLOCK-END
