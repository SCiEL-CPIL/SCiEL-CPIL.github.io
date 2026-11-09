"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] to improve numerical stability for high-degree polynomials
    z = x / 2.0
    
    # Construct a degree-9 polynomial using all 10 parameters
    # y = p0 + p1*z + p2*z^2 + ... + p9*z^9
    
    # Using Horner's method or vectorized polynomial evaluation
    # np.polyval expects coefficients in descending order [p9, p8, ..., p0]
    # But params are passed in arbitrary order, so we just build the sum explicitly
    # to avoid confusion with coefficient ordering if BFGS permutes them (it doesn't, but clarity is good).
    
    # Vectorized evaluation:
    # z^0, z^1, ..., z^9
    powers = np.stack([z**i for i in range(10)], axis=-1)
    
    # params is shape (10,), powers is shape (N, 10)
    # Result is shape (N,)
    return np.sum(powers * params, axis=1)
# EVOLVE-BLOCK-END
