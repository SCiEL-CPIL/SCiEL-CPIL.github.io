"""Symbolic regression with 1 input: bounded basis expansion for robust BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a bounded, well-conditioned basis of 10 functions.
    x is assumed to be in [-3, 3]. We scale to [-1, 1] for numerical stability.
    The 10 basis functions cover polynomials (up to degree 4), trigonometric,
    Gaussian, and tanh, all bounded or well-behaved on [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    xs = x / 3.0  # scale to [-1, 1]

    # Build basis functions (each shape (n,))
    b0 = np.ones_like(xs)
    b1 = xs
    b2 = xs ** 2
    b3 = xs ** 3
    b4 = xs ** 4
    b5 = np.sin(np.pi * xs)
    b6 = np.cos(np.pi * xs)
    b7 = np.sin(2.0 * np.pi * xs)
    b8 = np.exp(-xs ** 2)
    b9 = np.tanh(xs)

    # Linear combination with the 10 params
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)

    return y
# EVOLVE-BLOCK-END
