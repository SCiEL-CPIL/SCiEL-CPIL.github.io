"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a polynomial basis with learnable coefficients.
    params: array of length 10.
    We use all 10 parameters as coefficients for a 9th-degree polynomial in x.
    To ensure stability and allow the optimizer to converge, we scale x into [-1, 1]
    and use a scaled polynomial basis. This helps BFGS starting from ones.
    
    The basis functions are:
    1, x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9
    But to avoid numerical issues with high powers, we use centered and scaled x.
    Let z = x / 3.0  (so z is in [-1, 1])
    Then we use powers of z: 1, z, z^2, ..., z^9.
    
    This gives a rich family of functions that can approximate many smooth functions.
    """
    # Scale input to [-1, 1] for better conditioning
    z = x / 3.0
    
    # Build polynomial basis up to degree 9
    # Use Horner-like construction or just compute powers
    # To be safe and efficient, compute powers iteratively
    basis = np.ones_like(z)
    powers = [basis]
    for i in range(1, 10):
        powers.append(powers[-1] * z)
    
    # Stack into a matrix: each column is a basis function
    # shape: (n_points, 10)
    B = np.column_stack(powers)
    
    # Prediction: B @ params
    pred = B @ params
    
    return pred
# EVOLVE-BLOCK-END
