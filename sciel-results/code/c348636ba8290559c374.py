"""Symbolic regression with a diverse hybrid basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis for symbolic regression.
    
    The basis includes polynomial, trigonometric, exponential, and rational terms.
    This diversity allows the model to approximate a wider variety of target functions
    compared to a pure polynomial or Chebyshev basis.
    
    Basis functions:
    1. 1
    2. x
    3. x^2
    4. sin(x)
    5. cos(x)
    6. sin(2x)
    7. cos(2x)
    8. exp(-x)
    9. exp(-0.5x)
    10. 1/x
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 0.5) # Default to mid-range if invalid
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Avoid division by zero or near-zero for 1/x
    # x is in [0.1, 6], so 1/x is safe, but we clip just in case
    x_safe = np.clip(x, 1e-6, None)
    
    # Compute basis functions
    b0 = np.ones_like(x_safe)
    b1 = x_safe
    b2 = x_safe ** 2
    b3 = np.sin(x_safe)
    b4 = np.cos(x_safe)
    b5 = np.sin(2.0 * x_safe)
    b6 = np.cos(2.0 * x_safe)
    b7 = np.exp(-x_safe)
    b8 = np.exp(-0.5 * x_safe)
    b9 = 1.0 / x_safe
    
    # Stack into design matrix (N, 10)
    X = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction
    pred = X @ p
    
    # Ensure output is finite
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
