"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to approximately [-1, 1] to improve conditioning
    # Range [0.1, 6.0] -> center 3.05, half-range 2.95
    x_scaled = (x - 3.05) / 2.95
    
    # Compute basis functions: 1, x, x^2, ..., x^9
    # Using vectorized power
    # params[0] * 1 + params[1] * x_scaled + ... + params[9] * x_scaled^9
    
    # Efficiently compute powers
    # We can use np.vander or just iterative multiplication
    # np.vander(x_scaled, N=10, increasing=True) gives columns [1, x, x^2, ...]
    basis = np.vander(x_scaled, N=10, increasing=True)
    
    # Combine with parameters
    return basis @ params
# EVOLVE-BLOCK-END
