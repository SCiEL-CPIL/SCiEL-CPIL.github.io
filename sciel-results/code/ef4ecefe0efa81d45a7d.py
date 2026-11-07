"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [-1, 1] to improve numerical stability of polynomial basis
    # x range is [0.1, 6], center ~3.05, half-range ~2.95
    # u = (x - center) / scale
    center = 3.05
    scale = 2.95
    
    u = (x - center) / scale
    
    # Construct polynomial basis up to degree 9
    # u^0, u^1, ..., u^9
    # We can use np.vander or simple loop/multiplication
    # np.vander returns columns [u^N, ..., u^1, u^0]
    # We want [u^0, ..., u^9] to match params[0..9]
    
    # Using vander with increasing=False gives [u^9, ..., u^0]
    # So we can reverse the params or the basis
    # Let's just compute powers directly for clarity and control
    # u^0 = 1
    # u^1 = u
    # ...
    
    basis = np.ones_like(u)
    powers = [basis]
    for i in range(1, 10):
        basis = basis * u
        powers.append(basis)
        
    # Stack into a matrix of shape (N, 10)
    # Each column is u^i
    X_poly = np.column_stack(powers)
    
    # Prediction is X_poly @ params
    return X_poly @ params
# EVOLVE-BLOCK-END
