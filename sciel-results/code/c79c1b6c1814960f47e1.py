"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize input to [-1, 1] to keep basis functions well-conditioned
    t = x / 3.0
    
    # Define a diverse set of 10 basis functions
    # 1-5: Low-order polynomials (stable, captures trends)
    # 6: Hyperbolic tangent (captures saturation/non-linearity)
    # 7-8: Trigonometric (captures periodic features)
    # 9-10: Exponentials (captures growth/decay)
    
    B0 = np.ones_like(t)
    B1 = t
    B2 = t**2
    B3 = t**3
    B4 = t**4
    B5 = np.tanh(t)
    B6 = np.sin(t)
    B7 = np.cos(t)
    B8 = np.exp(t)
    B9 = np.exp(-t)
    
    # Stack basis functions into a matrix (N, 10)
    # Using a linear combination: y = sum(params[i] * B_i)
    # We use a dot product for efficiency
    
    # Note: We do not apply explicit scaling here initially, as the parameters
    # are free to adjust. However, if BFGS struggles, scaling might be needed.
    # Given the previous score was very high, the basis choice is likely the main issue.
    
    # Construct the design matrix
    # To avoid memory issues for large N, we compute the sum directly
    y = (params[0] * B0 +
         params[1] * B1 +
         params[2] * B2 +
         params[3] * B3 +
         params[4] * B4 +
         params[5] * B5 +
         params[6] * B6 +
         params[7] * B7 +
         params[8] * B8 +
         params[9] * B9)
         
    return y
# EVOLVE-BLOCK-END
