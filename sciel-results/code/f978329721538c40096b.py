"""Fourier series basis for symbolic regression on [-3, 3]."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [0, 1] for standard Fourier basis on [-3, 3]
    # Actually, use sin(k*pi*x/3) and cos(k*pi*x/3) for k=1..5
    # That gives 10 basis functions, matching 10 params
    
    x = np.asarray(x, dtype=np.float64)
    
    # Build Fourier basis: sin(k*pi*x/3) and cos(k*pi*x/3) for k=1..5
    # params[0] -> sin(pi*x/3)
    # params[1] -> cos(pi*x/3)
    # params[2] -> sin(2*pi*x/3)
    # params[3] -> cos(2*pi*x/3)
    # ... and so on
    
    k_vals = np.arange(1, 6)  # 1, 2, 3, 4, 5
    
    # Compute all basis values at once
    # shape: (n_points, 5) for sin, (n_points, 5) for cos
    arg = np.pi * np.outer(x, k_vals) / 3.0  # (n_points, 5)
    
    sin_basis = np.sin(arg)  # (n_points, 5)
    cos_basis = np.cos(arg)  # (n_points, 5)
    
    # params layout: [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5]
    # where a_k multiplies sin(k*pi*x/3) and b_k multiplies cos(k*pi*x/3)
    a_coeffs = params[0::2]  # params[0], params[2], params[4], params[6], params[8]
    b_coeffs = params[1::2]  # params[1], params[3], params[5], params[7], params[9]
    
    # Linear combination
    pred = sin_basis @ a_coeffs + cos_basis @ b_coeffs
    
    return pred
# EVOLVE-BLOCK-END
