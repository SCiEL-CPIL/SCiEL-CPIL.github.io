"""
Strategy: A neural network with a single hidden layer (tanh activation) as a
universal approximator, combined with a linear baseline for stability.

This is a structural departure from the previous trigonometric harmonic basis.
A 1-hidden-layer network with tanh is a universal approximator and can model
non-periodic, non-smooth, and complex non-linear dependencies that fixed-frequency
sin/cos bases struggle with. The linear baseline ensures the model can always
capture the mean trend and provides a stable starting point for BFGS.

Architecture:
  Hidden layer (2 units):
    h1 = tanh(w11*x + w12*y + w13*z + b1)
    h2 = tanh(w21*x + w22*y + w23*z + b2)
  
  Output:
    f(x, y, z) = c0 + c1*x + c2*y + c3*z + v1*h1 + v2*h2

Parameters (10 total):
  0: c0       (intercept)
  1: c1       (x coefficient in linear baseline)
  2: c2       (y coefficient in linear baseline)
  3: c3       (z coefficient in linear baseline)
  4: w11      (hidden weight: x -> h1)
  5: w12      (hidden weight: y -> h1)
  6: w13      (hidden weight: z -> h1)
  7: w21      (hidden weight: x -> h2)
  8: w22      (hidden weight: y -> h2)
  9: w23      (hidden weight: z -> h2)

Note: Biases b1, b2 and output weights v1, v2 are fixed to 1.0 to keep
within the 10-parameter budget. The linear baseline absorbs constant offsets
and linear trends, so the hidden layer can focus on non-linear structure.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a linear baseline plus a 2-unit tanh hidden layer.
    """
    p = params.flatten()
    
    c0, c1, c2, c3 = p[0], p[1], p[2], p[3]
    w11, w12, w13 = p[4], p[5], p[6]
    w21, w22, w23 = p[7], p[8], p[9]
    
    # Linear baseline
    linear = c0 + c1 * x + c2 * y + c3 * z
    
    # Hidden layer with tanh activation
    # Fixed biases b1=1.0, b2=1.0 and output weights v1=1.0, v2=1.0
    h1 = np.tanh(w11 * x + w12 * y + w13 * z + 1.0)
    h2 = np.tanh(w21 * x + w22 * y + w23 * z + 1.0)
    
    # Output
    return linear + h1 + h2
