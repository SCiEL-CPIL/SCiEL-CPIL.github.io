import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input from [-3, 3] to [-1, 1] for standard Chebyshev polynomials
    t = x / 3.0
    # Evaluate the Chebyshev series sum_{i=0}^{9} params[i] * T_i(t)
    # chebval(t, c) computes this efficiently
    return np.polynomial.chebyshev.chebval(t, params)
