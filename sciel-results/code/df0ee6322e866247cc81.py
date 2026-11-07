"""Symbolic regression with Fourier series basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with 10 parameters.
    
    Uses a Fourier series basis consisting of sine and cosine terms with
    frequencies scaled to the domain [0.1, 6]. This provides a globally
    smooth, orthogonal representation that avoids numerical instability of
    high-degree polynomials and locality constraints of RBFs.
    
    params[0:10] are coefficients for 10 basis functions:
    0: 1 (bias/DC term)
    1: cos(2*pi*1*x/6)
    2: sin(2*pi*1*x/6)
    3: cos(2*pi*2*x/6)
    4: sin(2*pi*2*x/6)
    5: cos(2*pi*3*x/6)
    6: sin(2*pi*3*x/6)
    7: cos(2*pi*4*x/6)
    8: sin(2*pi*4*x/6)
    9: cos(2*pi*5*x/6)
    """
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.clip(x, 1e-10, None)
    
    # Domain length for frequency scaling
    L = 6.0
    
    # Build Fourier basis
    b0 = np.ones_like(x_safe)
    
    # Frequency k=1
    b1 = np.cos(2 * np.pi * 1 * x_safe / L)
    b2 = np.sin(2 * np.pi * 1 * x_safe / L)
    
    # Frequency k=2
    b3 = np.cos(2 * np.pi * 2 * x_safe / L)
    b4 = np.sin(2 * np.pi * 2 * x_safe / L)
    
    # Frequency k=3
    b5 = np.cos(2 * np.pi * 3 * x_safe / L)
    b6 = np.sin(2 * np.pi * 3 * x_safe / L)
    
    # Frequency k=4
    b7 = np.cos(2 * np.pi * 4 * x_safe / L)
    b8 = np.sin(2 * np.pi * 4 * x_safe / L)
    
    # Frequency k=5
    b9 = np.cos(2 * np.pi * 5 * x_safe / L)
    
    # Stack into matrix: shape (n, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
